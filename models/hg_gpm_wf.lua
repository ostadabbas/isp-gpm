 -- From https://github.com/anewell/pose-hg-train
 -- with filter version, sigmoid normalization
require 'cudnn'
require 'nn'

paths.dofile('layers/Residual.lua')

local function hourglass_wf(n, f, inp) -- f num input and output channels, inp the input tensor?  n, how many pool layers in one hg
    -- f features numbers
    -- inp input image
    -- Upper branch , encoder 3/4  img covered , original to 256 fts
    local up1 = inp
    for i = 1,opt.nModules do up1 = Residual(f,f)(up1) end -- how many residual before hg  f in f out channels batchNorm in
-- residual seems no size reduce , each hourglass with one resi
    -- Lower branch
    local low1 = cudnn.SpatialMaxPooling(2,2,2,2)(inp)  -- -> 64
    --local low  = cudnn.SpatialFullConvolution()
    for i = 1,opt.nModules do low1 = Residual(f,f)(low1) end
    local low2

    if n > 1 then low2 = hourglass_wf(n-1,f,low1) -- return the combined one, the central part as a whole
    else
        low2 = low1
        --for i = 1,opt.nModules do low2 = Residual(f*2, f*2)(low2) end  -- high mid
        for i = 1,opt.nModules do low2 = Residual(f, f)(low2) end   -- original
    end
    local low3 = low2
    -- middle layer 2 f
    --if n>1 then
    --    for i = 1,opt.nModules do low3 = Residual(f,f)(low3) end
    --else
    --    for i = 1, opt.nModules do low3 = Residual(f*2, f*2)(low3) end
    --end
    -- original version
    for i = 1,opt.nModules do low3 = Residual(f,f)(low3) end
    --local up2 = nn.SpatialUpSamplingNearest(2)(low3)  -- do the up sampling
    local up2 = deconvBlock(f,f)(low3)
    --local up2W = nn.SpatialUpSamplingNearest(2)(low3) -- 2 is scale
    --local up2w = nn.SpatialBatchNormalization(f)(deconvBlock(f,f)(low3)) -- wf for skip
    --local up2w =  cudnn.Sigmoid()(nn.SpatialUpSamplingBilinear(2)(low3))  -- sig nomral
    local up2w =  cudnn.Sigmoid()(deconvBlock(f,f)(low3))
    --up1:CMul(up2W)
    up1 = nn.CMulTable()({up1, up2w})
    up1 = cudnn.SpatialConvolution(opt.nFeats,opt.nFeats,1,1,1,1,0,0)(up1)  -- reshuffle
    -- Bring two branches together
    return nn.CAddTable()({up1,up2})    -- summation of ts up1, up2
end

local function lin(numIn,numOut,inp)
    -- Apply 1x1 convolution, stride 1, no padding, with relu
    local l = cudnn.SpatialConvolution(numIn,numOut,1,1,1,1,0,0)(inp)    -- input planes and output planes
    return cudnn.ReLU(true)(nn.SpatialBatchNormalization(numOut)(l))
end
 local function linNorm(numIn,numOut,inp)
    -- Apply 1x1 convolution, stride 1, no padding, with relu
    local l = cudnn.SpatialConvolution(numIn,numOut,1,1,1,1,0,0)(inp)    -- input planes and output planes
    return nn.SpatialBatchNormalization(numOut)(l)
end


function createModelG()
-- output a table, each one is a heatmap of stage n.  SpatialCrossEntropyCriterion()
    local inp = nn.Identity()() -- a graph node nFts to 128
    local jMap = nn.Identity()()
--
--    -- Initial processing of the image inSize ch x h x w
--    local cnv1_ = cudnn.SpatialConvolution(opt.inSize[1],64,7,7,2,2,3,3)(inp)           -- 128
--    local cnv1 = cudnn.ReLU(true)(nn.SpatialBatchNormalization(64)(cnv1_))
--    local r1 = Residual(64,128)(cnv1)
--    local pool = cudnn.SpatialMaxPooling(2,2,2,2)(r1)                       -- 64
--    local r4 = Residual(128,128)(pool)
--    local r5 = Residual(128,opt.nFeats)(r4)
    -- 128 img two residual
    local inpCmb = nn.JoinTable(2)({inp, jMap}) -- along 2nd ch dim assume 4D
    local r4 = Residual(opt.inSize[1] + #opt.jointsIx, 128)(inpCmb)
    local r5 = Residual(128, opt.nFeats)(r4)    -- nfeats 256

    local out = {}
    local inter = r5
    --local inter = nn.JoinTable(2)({r5,})

    for i = 1,opt.nStack do -- to 3 stack
        local hg = hourglass_wf(5, opt.nFeats, inter) -- 5 scale level
        -- Residual layers at output resolution
        local ll = hg
        for j = 1,opt.nModules do ll = Residual(opt.nFeats,opt.nFeats)(ll) end
        -- Linear layer to produce first set of predictions
        ll = lin(opt.nFeats,opt.nFeats,ll)  -- in place spacialconv
        -- conv bn relu ...  image should be good no huge error.
        -- Predicted heatmaps
        local tmpOut = cudnn.SpatialConvolution(opt.nFeats,opt.nOutChannels,1,1,1,1,0,0)(ll)
        table.insert(out, tmpOut)

        -- Add predictions back
        if i < opt.nStack then
            local tmpOutCmp = nn.JoinTable(2)({tmpOut, jMap})
            --local ll2 = cudnn.SpatialConvolution(opt.nFeats,opt.nFeats,1,1,1,1,0,0)(ll) -- add a ll2 in between
            --local llW = cudnn.SpatialConvolution(opt.nFeats, opt.nOutChannels + #opt.jointsIx,3,3,1,1,1,1)(ll2)    -- a little margin, weights out the new out
            --local llw = linNorm(opt.nFeats, opt.nOutChannels + #opt.jointsIx, ll) -- spatial norm one, not stable I think, no necessary, as upCh already redistribute all the channels, but inter should be remixed
            local ll_ = cudnn.SpatialConvolution(opt.nFeats,opt.nFeats,1,1,1,1,0,0)(ll)
            --tmpOutCmp:CMul(llW) -- weighted out
            --tmpOutCmp = nn.CMulTable()({tmpOutCmp, llw})
            local tmpOut_ = cudnn.SpatialConvolution(opt.nOutChannels + #opt.jointsIx,opt.nFeats,1,1,1,1,0,0)(tmpOutCmp)    -- redistribute
            local interC11 = cudnn.SpatialConvolution(opt.nFeats,opt.nFeats,1,1,1,1,0,0)(inter)
            inter = nn.CAddTable()({interC11, ll_, tmpOut_})
        end
    end
    -- Final model
    --local model = nn.gModule({inp}, out)    -- out a table, {inp} table of input
    local model = nn.gModule({inp, jMap}, out)  -- two input
    print('Stacked hourglass initialized from scratch')
    return model
end
