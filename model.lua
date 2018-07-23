require 'nn'
require 'cunn'
require 'optim'
require 'cudnn'
-- nngraph in main called
local function crit()  -- selective structure
    if opt.criterion == 'ABS' then
        return nn.AbsCriterion()  -- cudnn no AbsCriterion
    elseif opt.criterion == 'MSE' then
        return nn.MSECriterion()
    else
        print('no such criterion')
        return nil
    end
end
-- Criterion
criterionAE = nn.ParallelCriterion()   -- weighted sum of other criterions

if opt.cGAN then
    criterionDisc = nn.ParallelCriterion()
end

for st = 1, opt.nStack do -- 8 hg number add all together
    --criterion:add(cudnn.SpatialCrossEntropyCriterion()) -- input nBch x nCls x h x w , average cross entropy
    criterionAE:add(crit())
    if opt.cGAN then
        criterionDisc:add(nn.BCECriterion())
    end
end
-- all classes 0 but the true one 1, add sum log(a) together.
-- Create Network
--    If preloading option is set, preload weights from existing models appropriately
--    If model has its own criterion, override.
if opt.retrainG ~= 'none' then -- check only retrainG
    assert(paths.filep(opt.retrainG), 'File not found: ' .. opt.retrainG)
    print('Loading model from file: ' .. opt.retrainG);
    netG = loadDataParallel(opt.retrainG, opt.nGPU)
else
    paths.dofile('models/' .. opt.netType .. '.lua')    -- hourglass
    print('=> Creating model from file: models/' .. opt.netType .. '.lua')
    netG = createModelG(opt.nGPU ) -- for the model creation code, check the models/ folder
    if opt.backend == 'cudnn' then
        require 'cudnn'
        cudnn.convert(netG, cudnn) --
    elseif opt.backend == 'cunn' then
        require 'cunn'
        netG = netG:cuda()
    elseif opt.backend ~= 'nn' then
        error'Unsupported backend'
    end
end
print('=> netG')
print(netG)
print('=> Criterion')
print(criterionAE)
-- Convert model to CUDA
print('==> Converting model and criterion to CUDA')
netG:cuda()
criterionAE:cuda()

-- netD
if opt.cGAN then
    if opt.retrainD ~= 'none' then -- check only retrainG
        assert(paths.filep(opt.retrainD), 'File not found: ' .. opt.retrainD)
        print('Loading model from file: ' .. opt.retrainD);
        netD = loadDataParallel(opt.retrainD, opt.nGPU)
    else
        --paths.dofile('models/' .. opt.netType .. '.lua')    -- hourglass
        paths.dofile('models/cGAN_model.lua')
        print('=> Creating model from file: modelss/cGAN_model.lua')
        netD_module = defineD_n_layers(opt.inSize[1]+ #opt.jointsIx, opt.inSize[1], opt.ndf, opt.D_nLayers) -- for the model creation code, check the models/ folder
        netD = nn.MapTable()
        netD:add(netD_module)   -- as the single module
        if opt.backend == 'cudnn' then  -- only for new creation
            require 'cudnn'
            cudnn.convert(netD, cudnn) --
        elseif opt.backend == 'cunn' then
            require 'cunn'
            netD = netD:cuda()
        elseif opt.backend ~= 'nn' then
            error'Unsupported backend'
        end
    end
    print('=> netD')
    print(netD)
    print('=> Criterion disc')
    print(criterionDisc)
    -- Convert model to CUDA
    print('==> Converting model and criterion to CUDA')
    netD:cuda()
    criterionDisc:cuda()
end

cudnn.fastest = true
cudnn.benchmark = true

collectgarbage()
