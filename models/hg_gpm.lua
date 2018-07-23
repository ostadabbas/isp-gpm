-- gpm hourglass with fully convolution configuration and conditional LDPD
-- Author: Shuangjun Liu (NEU/AClab)
-- Original work: From https://github.com/anewell/pose-hg-train

paths.dofile('layers/Residual.lua')

local function hourglassDcv(n, f, inp) -- f num input and output channels, inp the input tensor?  n, how many pool layers in one hg
    -- f features numbers
    -- inp input image
    -- Upper branch , encoder 3/4  img covered , original to 256 fts
    local up1 = inp
    for i = 1,opt.nModules do up1 = Residual(f,f)(up1) end -- how many residual before hg
    -- Lower branch
    local low1
    if opt.use_fcn ==1 then     -- fc style
        print('spatial Conv downsampling layer employed')
        -- spatial normal
        low1 = cudnn.SpatialBatchNormalization(f)(inp)
        low1 = nn.LeakyReLU(0.2, true)(low1)
        -- leakyRelu
        low1 = cudnn.SpatialConvolution(f, f, 4, 4, 2, 2, 1, 1)(low1)
    else
        print('Max pooling downsampling layer employed')
        low1 = cudnn.SpatialMaxPooling(2,2,2,2)(inp)
    end

    --local low  = cudnn.SpatialFullConvolution()
    for i = 1,opt.nModules do low1 = Residual(f,f)(low1) end
    local low2

    if n > 1 then low2 = hourglassDcv(n-1,f,low1) -- return the combined one, the central part as a whole
    else
        low2 = low1
        --for i = 1,opt.nModules do low2 = Residual(f*2, f*2)(low2) end  -- high mid
        for i = 1,opt.nModules do low2 = Residual(f, f)(low2) end   -- original
    end
    local low3 = low2

    for i = 1,opt.nModules do low3 = Residual(f,f)(low3) end
    --local up2 = nn.SpatialUpSamplingNearest(2)(low3)  -- do the up sampling
    local up2
    if opt.use_fcn ==1 then
        up2 = deconvBlock(f,f)(low3)
    else
        up2 = nn.SpatialUpSamplingNearest(2)(low3)
    end

    -- Bring two branches together
    return nn.CAddTable()({up1,up2})    -- summation of ts up1, up2
end

local function lin(numIn,numOut,inp)
    -- Apply 1x1 convolution, stride 1, no padding, with relu
    local l = cudnn.SpatialConvolution(numIn,numOut,1,1,1,1,0,0)(inp)    -- input planes and output planes
    return cudnn.ReLU(true)(nn.SpatialBatchNormalization(numOut)(l))
end

function createModelG()
-- output a table, each one is a heatmap of stage n.  SpatialCrossEntropyCriterion()
    local inp = nn.Identity()() -- a graph node nFts to 128
    local jMap = nn.Identity()()

    local inpCmb = nn.JoinTable(2)({inp, jMap}) -- along 2nd ch dim assume 4D
    local r4 = Residual(opt.inSize[1] + #opt.jointsIx, 128)(inpCmb)
    local r5 = Residual(128, opt.nFeats)(r4)

    local out = {}
    local inter = r5
    --local inter = nn.JoinTable(2)({r5,})

    for i = 1,opt.nStack do -- to 3 stack
        local hg = hourglassDcv(5,opt.nFeats,inter) -- 5 scale level
        -- Residual layers at output resolution
        local ll = hg
        for j = 1,opt.nModules do ll = Residual(opt.nFeats,opt.nFeats)(ll) end
        -- Linear layer to produce first set of predictions
        ll = lin(opt.nFeats,opt.nFeats,ll)  -- in place spacialconv

        -- Predicted heatmaps this part should be regulated I think
        local tmpOut = cudnn.SpatialConvolution(opt.nFeats,opt.nOutChannels,1,1,1,1,0,0)(ll)
        if opt.ifTanh ==1 then
            tmpOut = nn.Tanh()(tmpOut)    -- regulate the range
        end
        table.insert(out, tmpOut)

        -- Add predictions back
        if i < opt.nStack then
            local tmpOutCmp = nn.JoinTable(2)({tmpOut, jMap})
            local ll_ = cudnn.SpatialConvolution(opt.nFeats,opt.nFeats,1,1,1,1,0,0)(ll)
            local tmpOut_ = cudnn.SpatialConvolution(opt.nOutChannels + #opt.jointsIx,opt.nFeats,1,1,1,1,0,0)(tmpOutCmp)
            inter = nn.CAddTable()({inter, ll_, tmpOut_})
        end
    end
    -- Final model
    local model = nn.gModule({inp, jMap}, out)  -- two input
    print('Stacked hourglass initialized from scratch')
    return model
end
