-- GPM training session

require 'optim'  -- I like to put it make it clear to user which func is called in
require 'util.lua'
-- Setup a reused optimization state. If needed, reload from disk
display = require 'display'

-- load first netG
if opt.optimStateG ~= 'none' then
    assert(paths.filep(opt.optimStateG), 'File not found: ' .. opt.optimStateG)   -- filep if exist a file
    print('Loading optimState from file: ' .. opt.optimStateG)
    optimStateG = torch.load(opt.optimStateG)
    --optimStateG.learningRate = opt.LR -- update LR, I think we should keep original
end
if not optimStateG then -- keep same optimState at this time
    if opt.optType == 'Adam' then
        print('Adam optimization')
        optimStateG = {
            learningRate = opt.LR_Adam,
            beta1 = opt.beta1,
        }
    else
        optimStateG = {
        learningRate = opt.LR,
        learningRateDecay = 0.0,
        momentum = opt.momentum,
        weightDecay = opt.weightDecay,
        dampening = 0.0,
        alpha = opt.alpha,  -- rmsprop  the gradient coeff?
        epsilon = opt.epsilon
        }
    end
end -- alpha

-- netD
if opt.cGAN then
    if opt.optimStateD ~= 'none' then
        assert(paths.filep(opt.optimStateD), 'File not found: ' .. opt.optimStateD)   -- filep if exist a file
        print('Loading optimState from file: ' .. opt.optimStateD)
        optimStateD = torch.load(opt.optimStateD)
        --optimStateD.learningRate = opt.LR -- update LR
    end
    if not optimStateD then -- for D net, doesn't hurt for this small memory
        if opt.optType == 'Adam' then
            print('Adam for netD')
            optimStateD = {
               learningRate = opt.LR_Adam,
               beta1 = opt.beta1,
            }
        else
            optimStateD = {
                learningRate = opt.LR,
                learningRateDecay = 0.0,
                momentum = opt.momentum,
                weightDecay = opt.weightDecay,
                dampening = 0.0,
                alpha = opt.alpha,  -- rmsprop  the gradient coeff?
                epsilon = opt.epsilon
            }
        end
    end -- alpha
end


-- Logger
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))  -- seg data,train log in place
local batchNumber
local lossG, lossD, lossL1, lossAll

-------------------------------------------------------------------------------------------

-- GPU inputs (preallocate)
local inputs = torch.CudaTensor()   -- batch
local real_B = torch.CudaTensor() -- read in single batch, the target ground truth
local jMaps = torch.CudaTensor()
-- put in if not in the file scope then,
local real_A = torch.CudaTensor() -- input with jMaps
local real_Bs = {}      -- save all true labels
local fake_Bs = {}  -- list output  batch list
local real_AB = torch.CudaTensor()  -- list output for
local fake_AB = torch.CudaTensor()
local real_ABs = {} -- list output  -- duplicated list  for criterion input
local fake_ABs = {} -- list input   -- same real_A but stages fake
local bMap_Rs = {} -- real_ABs through netD
local bMap_Fs = {} -- fake_ABs through netD
--local bMap_FRs = {} -- fake with real labels for G loss
local real_label = 1
local fake_label = 0
local errD, errG, errL1, errAll= 0, 0, 0, 0-- errD for disc error, errG for cheating D loss,
local jMap_sum = torch.Tensor()  -- for single jMap sum to save skeleton image

local timer = torch.Timer()
local dataTimer = torch.Timer()

-- display plot config
local plot_config = {
  title = "Loss over time",
  labels = {"epoch", unpack(opt.display_plot)},
  ylabel = "loss",
}

-- display plot vars
local plot_data = {}
local plot_win

local parametersG, gradParametersG = netG:getParameters() -- flat tensors(learnable parameters like w,b
local parametersD, gradParametersD
if opt.cGAN then
    parametersD, gradParametersD = netD:getParameters()
end



local fDx = function(x)
    -- double FB process to update both real and fake, make fake one 1
    netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

    gradParametersD:zero()

    -- Real
    bMap_Rs = netD:forward(real_ABs) -- table back
    local label = torch.FloatTensor(bMap_Rs[1]:size()):fill(real_label)
    local labels = {}
    if opt.nGPU >0 then
    	label = label:cuda()
    end
    for i = 1, opt.nStack do
        table.insert(labels, label)
    end
    local errD_real = criterionDisc:forward(bMap_Rs, labels) -- sigma p*log(p_)  q*log(q_)  that is 1*log(real)  (1-0)log(1-p_) pre real how much loss + pre fake how much loss
    local df_do = criterionDisc:backward(bMap_Rs, labels)   -- difference at output
    netD:backward(real_ABs, df_do)
    -- Fake
    bMap_Fs = netD:forward(fake_ABs)   -- predict table
    local label = torch.FloatTensor(bMap_Fs[1]:size()):fill(fake_label)
    local labels = {} -- true table
    if opt.nGPU>0 then
    	label = label:cuda()
    end
    for i = 1, opt.nStack do
        table.insert(labels, label)
    end
    local errD_fake = criterionDisc:forward(bMap_Fs, labels) -- sigma p*log(p_)  q*log(q_)  that is 1*log(real)  (1-0)log(1-p_) pre real how much loss + pre fake how much loss
    local df_do = criterionDisc:backward(bMap_Fs, labels)
    netD:backward(fake_ABs, df_do)
    errD = (errD_real + errD_fake)/2
    return errD, gradParametersD
end

local fGx = function(x)
    netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

    gradParametersG:zero()

    -- GAN loss
    --local df_dg = torch.zeros(fake_Bs[1]:size())
    local df_dgs = {} -- keep list of df_dgs
    local df_dABs = {} -- the df to AB input in netD


    if opt.cGAN  then

        local label = torch.FloatTensor(bMap_Fs[1]:size()):fill(real_label)
        local labels = {} -- true table
        if opt.nGPU>0 then
            label = label:cuda()
        end
        for i = 1, opt.nStack do
            table.insert(labels, label)
        end

       errG = criterionDisc:forward(bMap_Fs, labels)  -- loss to fool the netD
       local df_dos = criterionDisc:backward(bMap_Fs, labels)   -- grad to all outputs
        df_dABs = netD:updateGradInput(fake_ABs,df_dos) -- larger dim grads -- no value gotten here
       --df_dABs = netD:updateGradInput(fake_ABs, df_do):narrow(2,fake_AB:size(2)-output_nc+1, output_nc)  -- only last few channels are for netG, previous are pure steady input. fake_AB concat gradient w.r.t own input
        -- the grad pass back through netD, to input, which is the output portal of netG
        -- cut list to df_dg

    else
        errG = 0
    end

    -- unary loss
    --local df_do_AE = torch.zeros(fake_B:size())
    local df_do_AEs = {}

    if opt.use_L1==1 then
       errL1 = criterionAE:forward(fake_Bs, real_Bs)    -- should be scalar
       df_do_AEs = criterionAE:backward(fake_Bs, real_Bs)   -- grad wrt G output, should be ch3
    else
        errL1 = 0
    end

    -- grad tbl for  netG
    df_dgs = {}

    for i = 1, opt.nStack do
        if opt.use_L1==1 then
            df_dgs[i]=df_dABs[i]:narrow(2,fake_ABs[1]:size(2)-opt.inSize[1]+1, opt.inSize[1]) + df_do_AEs[i]:mul(opt.lambda)       -- output_nc here same as inputs
        else
            df_dgs[i]=df_dABs[i]:narrow(2,fake_ABs[1]:size(2)-opt.inSize[1]+1, opt.inSize[1])
        end
    end

    netG:backward({inputs, jMaps}, df_dgs) -- forward called before but only once

    return errG + errL1 * opt.lambda, gradParametersG   -- I add errL1 to it
end


function trainBatch_cGAN(inputsCPU, labelsCPU, jMapsCPU)  -- inputs images, labels of seg, the choosen indices, what is the instances CPU? last two not used. end call visible in main thread
    print("Into trainBatch function now")
    cutorch.synchronize()
    collectgarbage()
    local dataLoadingTime = dataTimer:time().real
    timer:reset()

    -- transfer over to GPU,
    inputs:resize(inputsCPU:size()):copy(inputsCPU) -- copy to cuda.tensor
    jMaps:resize(jMapsCPU:size()):copy(jMapsCPU)
    real_B:resize(labelsCPU:size()):copy(labelsCPU)

    -- create real and fake
    real_A = torch.cat(inputs, jMaps, 2)
    -- real_B already in
    real_AB = torch.cat(real_A, real_B, 2)
    real_ABs = {}    -- empty it
    for i = 1, opt.nStack do
        table.insert(real_ABs, real_AB)
    end

    real_Bs = {}
    for i = 1, opt.nStack do
        table.insert(real_Bs, real_B)
    end

    -- create fake
    fake_Bs = netG:forward({inputs, jMaps})  -- a list
    fake_ABs = {}
    for i = 1, opt.nStack do
        table.insert(fake_ABs, torch.cat(real_A, fake_Bs[i], 2))
    end

    if opt.cGAN then
        if opt.optType == 'Adam' then
            optim.adam(fDx, parametersD, optimStateD)
        else
            optim.rmsprop(fDx, parametersD, optimStateD)
        end
    end

    if opt.optType == 'Adam' then
        optim.adam(fGx,parametersG, optimStateG)
    else
        optim.rmsprop(fGx, parametersG, optimStateG)
    end

    -- err updated in fDx and fGx
    errAll = errG + errD + errL1 * opt.lambda
    local str = string.format("errG error %.2f, errD error, %.2f, L1 error %.2f",  errG, errD, errL1)

    -- global epoch can be accessed
    cutorch.synchronize()
    batchNumber = batchNumber + 1
    lossG = lossG + errG
    lossD = lossD + errD
    lossL1 = lossL1 + errL1
    lossAll = lossAll + errAll
    --print(('Epoch: [%d][%d/%d] Time: %.3f, Err: %.2f \t %s, \t LR: %.0e, \t DataLoadingTime %.3f'):format(
    --    epoch, batchNumber, opt.epochSize, timer:time().real, errL1, str,
    --    optimStateG.learningRate, dataLoadingTime))
    print(('Epoch: [%d][%d/%d] Time: %.3f, Total error, %.2f, %s, \t LR_G: %.0e, \t DataLoadingTime %.3f'):format(
    epoch, batchNumber, opt.epochSize, timer:time().real, errAll, str,
    optimStateG.learningRate, dataLoadingTime))

    trainEpochLogger:add{
        ['BatchNumber'] = string.format("%d", batchNumber),
        ['ErrG'] = string.format("%.8f", errG),
        ['ErrD'] = string.format("%.8f", errG),
        ['ErrL1'] = string.format("%.8f", errL1),
        ['ErrAll'] = string.format("%.8f", errAll),
        ['LR'] = string.format("%.0e", optimStateG.learningRate)
    }
    if 0 == batchNumber%opt.serUpdtRt then
        -- only save first image, last stage
        jMap_sum = jMaps[1]:clone()
        --print('after get jMap[1],j_sum size is', jMap_sum:size() )
        jMap_sum = jMap_sum:sum(1):repeatTensor(3,1,1)    -- fake 3 channel
        --print('jMap_sum size', jMap_sum:size())
        --print('inputs[1] size', inputs[1]:size())
        display.images({ deNormImg(inputs[1]), deNormImg(real_B[1]), jMap_sum, deNormImg(fake_Bs[opt.nStack][1])}, { win = opt.display_id, title ='GPM transform', labels = { 'origin', 'target', 'placeHolder', 'output'}})
        local loss = {errG=errG and errG or -1, errD=errD and errD or -1, errL1=errL1 and errL1 or -1}
        --local curItInBatch = ((batchNumber-1) / opt.batchSize)
        local plot_vals = {epoch + batchNumber/opt.epochSize}
        for k, v in ipairs(opt.display_plot) do
            if loss[v] ~= nil then
                plot_vals[#plot_vals + 1] = loss[v]
            end
        end
        table.insert(plot_data, plot_vals)
        plot_config.win = plot_win    -- plot window
        plot_win = display.plot(plot_data, plot_config)
    end
    -- save display
    dataTimer:reset()
end
-- Used by train() to train a single batch after the data is loaded. it should work but why not see the global ?
function trainBatch(inputsCPU, labelsCPU, jMapsCPU)  -- inputs images, labels of seg, the choosen indices, what is the instances CPU? last two not used. end call visible in main thread
    print("Into trainBatch function now")
    cutorch.synchronize()
    --print("cutorch synchronization completed")
    collectgarbage()
    local dataLoadingTime = dataTimer:time().real
    timer:reset()

    -- transfer over to GPU,
    inputs:resize(inputsCPU:size()):copy(inputsCPU) -- copy to cuda.tensor
    jMaps:resize(jMapsCPU:size()):copy(jMapsCPU)
    real_B:resize(labelsCPU:size()):copy(labelsCPU)


    local err, fake_B, target

    feval = function(x)
        netG:zeroGradParameters()
        fake_B = netG:forward({ inputs, jMaps}) -- table of nStack outputs

        target = {}  -- real images
        for st = 1, opt.nStack do
            table.insert(target, real_B)
        end

        err = criterionAE:forward(fake_B, target)    -- outputs targets all tables
        local gradOutputs = criterionAE:backward(fake_B, target)

        netG:backward({ inputs, jMaps}, gradOutputs)    -- gradOutputs should be list too
        return err, gradParametersG
    end

    optim.rmsprop(feval, parametersG, optimStateG)

    local str = string.format("Image %s error is %.2f", opt.criterion, err)

    -- global epoch can be accessed
    cutorch.synchronize()
    batchNumber = batchNumber + 1
    lossG = lossG + err
    print(('Epoch: [%d][%d/%d] Time: %.3f, Err: %.2f \t %s, \t LR: %.0e, \t DataLoadingTime %.3f'):format(
        epoch, batchNumber, opt.epochSize, timer:time().real, err, str,
        optimStateG.learningRate, dataLoadingTime))
    trainEpochLogger:add{
        ['BatchNumber'] = string.format("%d", batchNumber),
        ['Error'] = string.format("%.8f", err),
        ['LR'] = string.format("%.0e", optimStateG.learningRate)
    }
    if 0 == batchNumber%opt.serUpdtRt then
        display.images({ deNormImg(inputs[1]), deNormImg(real_B[1]), torch.Tensor(opt.inSize[1], opt.inSize[2], opt.inSize[3]):zero(), deNormImg(fake_B[opt.nStack][1])}, { win = idWinTr, title ='GPM transform', labels = { 'origin', 'target', 'placeHolder', 'output'}})
    end

    dataTimer:reset()
end


-- put it here, other func should be known before hand
function train()
    print('==> doing epoch on training data:')
    print("==> online epoch # " .. epoch)
    trainEpochLogger = optim.Logger(paths.concat(opt.save, ("epoch_%d_train.log"):format(epoch))) -- in place epoch_1_train.log

    batchNumber = 0
    cutorch.synchronize()   -- wait all cpu finish maybe here is the problem parts, cpu loading not completed

    -- set to training mode
    netG:training()
    netG:cuda()

    if opt.cGAN then
        netD:training()
        netD:cuda()
    end

    local tm = torch.Timer()
    lossG = 0
    lossD = 0
    lossL1 = 0
    lossAll = 0

    local trainFunc = nil
    if opt.cGAN then    -- give right train func
        trainFunc = trainBatch_cGAN
    else
        trainFunc = trainBatch
    end
    -- update should be updated by upper value ,
    for i=1,opt.epochSize do   -- 2000
        -- queue jobs to data-workers
        donkeys:addjob(
            -- the job callback (runs in data-worker thread)
            function()
                local inputs, labels, jMaps = trainLoader:sample(opt.batchSize)   -- batchsize 6 should be done quickly
                --print('sample gotten!')
                return inputs, labels, jMaps    -- no indices
            end,
            trainFunc  -- can feed in func name directly
        )
    end
    donkeys:synchronize()
    cutorch.synchronize()

    -- Performance measures
    lossG = lossG / opt.epochSize     -- each epoch add a loss
    lossD = lossD / opt.epochSize
    lossL1 = lossL1  / opt.epochSize
    lossAll = lossAll /opt.epochSize
    --lossAll = lossG + lossD + lossL1    -- add weight here to generate better result?

    -- global loss func
    table.insert(lossLi_tr, lossAll)        -- mat only keeps the lossAll
    trainLogger:add{        -- an optim logger
        ['epoch'] = epoch,
        ['lossG'] = lossG,
        ['lossG'] = lossD,
        ['lossL1'] = lossL1,
        ['lossALL'] = lossAll,
        ['LR'] = optimStateG.learningRate,
    }
    opt.plotter:add('LR', 'train', epoch, optimStateG.learningRate) -- add in train plotter--
    opt.plotter:add('lossG', 'train', epoch, lossG)
    opt.plotter:add('lossD', 'train', epoch, lossD)
    opt.plotter:add('lossL1', 'train', epoch, lossL1)
    opt.plotter:add('lossAll', 'train', epoch, lossAll)

    print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t' .. 'lossG: %.6f \t'.. 'lossD: %.6f \t'.. 'lossL1: %.6f \t'.. 'lossAll: %.6f \t',epoch, tm:time().real, lossG, lossD, lossL1, lossAll))

    print('\n')
    collectgarbage()
    netG:clearState()
    if opt.cGAN then
        netD:clearState()
    end
    if 1 == epoch then      -- at least save first epoch for test part
        saveDataParallel(paths.concat(opt.save, 'netG_' .. epoch .. '.t7'), netG)
        torch.save(paths.concat(opt.save, 'optimStateG_' .. epoch .. '.t7'), optimStateG)
        if opt.cGAN then
            saveDataParallel(paths.concat(opt.save, 'netD_' .. epoch .. '.t7'), netD)
            torch.save(paths.concat(opt.save, 'optimStateD_' .. epoch .. '.t7'), optimStateD)
        end
    end
end -- of train()