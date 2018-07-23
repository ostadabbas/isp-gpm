-- GPM test loop
-- Shuangjun Liu (NEU/ACLab)

if(opt.evaluate) then
    testLogger = optim.Logger(paths.concat(opt.save, 'evaluate.log'))
else
    testLogger = optim.Logger(paths.concat(opt.save, opt.testDir .. '.log'))
end

local batchNumber
local lossG, lossD, lossL1, lossAll
-- GPU inputs (preallocate)
local inputs = torch.CudaTensor()   -- batch
local real_B = torch.CudaTensor() -- read in single batch
local jMaps = torch.CudaTensor()
-- put in if not in the file scope then,
local real_A = torch.CudaTensor()
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
local errD, errG, errL1, errAll = 0, 0, 0 -- errD for disc error, errG for cheating D loss,

local jMap_sum  -- for single jMap sum to save skeleton image

local timer = torch.Timer()
function test()
    --local optimState    -- maybe for upvalue in test threads
    if(opt.evaluate) then
        print('==> Testing final predictions')
    else
        epochLoad = math.floor(epoch/10)*10 -- only save every 10 epoches
        if 0 == epochLoad then
            epochLoad =1
        end
        --optimState = torch.load(paths.concat(opt.save, 'optimState_' .. epochLoad .. '.t7')) -- not used in the process
        print('==> validation epoch # ' .. epoch)
    end

    batchNumber = 0
    cutorch.synchronize()
    timer:reset()

    -- set to evaluate mode
    netG:evaluate()    -- control the batchNorm dropout
    if opt.cGAN then
        netD:evaluate()
    end

    lossG = 0
    lossD = 0
    lossL1 = 0    -- upvalue
    lossAll = 0
    local testFunc = nil
    if opt.cGAN then
        testFunc = testBatch_cGAN
        print('take cGan test session')
    else
        testFunc = testBatch
        print('employ L1 only test session')
    end

    for i=1,nTest/opt.batchSize do -- nTest is set in data.lua
        local indexStart = (i-1) * opt.batchSize + 1    -- inorder load in image paths, why not the same?
        local indexEnd = math.min(nTest, indexStart + opt.batchSize - 1)
        donkeys:addjob(
            -- work to be done by donkey thread
            function()
                local inputs, labels, jMaps, indices = testLoader:get(indexStart, indexEnd)
                return inputs, labels, jMaps, indices
            end,
            testFunc
        )
    end

    donkeys:synchronize()   -- threads pool
    cutorch.synchronize()

    -- Performance measures:
    lossG = lossG / (nTest/opt.batchSize)
    lossD = lossD / (nTest/opt.batchSize)
    lossL1 = lossL1 / (nTest/opt.batchSize)     -- local loss
    lossAll = lossAll / (nTest/opt.batchSize)
    --lossAll = lossG + lossD + lossL1
    table.insert(lossLi_tst, lossAll)

    testLogger:add{
        ['epoch'] = epoch,
        ['lossG'] = lossG,
        ['lossG'] = lossD,
        ['lossL1'] = lossL1,
        ['lossALL'] = lossAll
    }
    if(not opt.evaluate) then
        opt.plotter:add('lossG', 'test', epoch, lossG)
        opt.plotter:add('lossD', 'test', epoch, lossD)
        opt.plotter:add('lossL1', 'test', epoch, lossL1)
        opt.plotter:add('lossAll', 'test', epoch, lossAll)
        print(string.format('Epoch: [%d] ', epoch))
    end

    print(string.format('[TESTING SUMMAR] Total Time(s): %.2f \t' .. 'lossG: %.6f \t'.. 'lossD: %.6f \t'.. 'lossL1: %.6f \t'.. 'lossAll: %.6f \t',timer:time().real, lossG, lossD, lossL1, lossAll))
    print('\n')
end -- of test()
-----------------------------------------------------------------------------

local inputs = torch.CudaTensor()
local real_B = torch.CudaTensor()
local jMaps = torch.CudaTensor()

function testBatch(inputsCPU, labelsCPU, jMapsCPU)
    batchNumber = batchNumber + opt.batchSize   -- actually imNum   the updated testBatch can have different images.

    inputs:resize(inputsCPU:size()):copy(inputsCPU)
    real_B:resize(labelsCPU:size()):copy(labelsCPU)
    jMaps:resize(jMapsCPU:size()):copy(jMapsCPU)

    local outputs = netG:forward({ inputs, jMaps})   -- a table
    -- num outputs = nStacks

    local target
    if(opt.nStack > 1) then
        target = {}
        -- Same ground truth for all 8 stacks
        for st = 1, opt.nStack do
            table.insert(target, real_B)
        end
    else
        target = real_B
    end
    local idStg = opt.nStack  -- which stage to save out
    if opt.evaluate then    -- only save when evaluate  opt.evaluate
        if batchNumber <= opt.numOutImgs  then
            -- save image to folder only last stage at this time
            --print('in save period')
            --print('the opt.ifAllStgs is', opt.ifAllStgs)
            for i= 1,opt.batchSize do
                image.save(paths.concat(opt.outImgsDir, 'test' .. '_sq' .. batchNumber - opt.batchSize + i .. '_st' .. idStg .. '_O.jpg'), deNormImg(outputs[opt.nStack][i]:clone())) -- last output image [nstacks][batchSize]
                if opt.ifAllStgs == 1 then
                    print('ifAllStgs is 1')
                    for j = 1, opt.nStack -1 do
                        image.save(paths.concat(opt.outImgsDir, 'test' .. '_sq' .. batchNumber - opt.batchSize + i .. '_st' .. j .. '_O.jpg'), deNormImg(outputs[j][i]:clone()))
                        print('save to', paths.concat(opt.outImgsDir, 'test' .. '_sq' .. batchNumber - opt.batchSize + i .. '_st' .. j .. '_O.jpg'))
                    end
                end
                image.save(paths.concat(opt.outImgsDir, 'test' .. '_sq' .. batchNumber - opt.batchSize + i .. '_st' .. idStg .. '_A.jpg'), deNormImg(inputs[i]:clone()))
                image.save(paths.concat(opt.outImgsDir, 'test' .. '_sq' .. batchNumber - opt.batchSize + i ..'_st' .. idStg ..  '_B.jpg'), deNormImg(real_B[i]:clone()))    -- deNormImg will change the value itself
            end
            -- display result
        end
    end

    -- Compute loss
    local err = criterionAE:forward(outputs, target)  -- nStack tables diff error

    cutorch.synchronize()
    lossL1 = lossL1 + err


    if(opt.evaluate) then   -- str for other standard, no suitable here
        print(string.format('Testing [%d/%d] \t Loss %.8f \t', batchNumber, nTest, err))
    else
        print(string.format('Epoch: Testing [%d][%d/%d] \t Loss %.8f \t', epoch, batchNumber, nTest, err))
    end
end

function testBatch_cGAN(inputsCPU, labelsCPU, jMapsCPU)
    batchNumber = batchNumber + opt.batchSize   -- actually imNum

    inputs:resize(inputsCPU:size()):copy(inputsCPU)
    real_B:resize(labelsCPU:size()):copy(labelsCPU)
    jMaps:resize(jMapsCPU:size()):copy(jMapsCPU)

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

    -- errD
    -- Real
    bMap_Rs = netD:forward(real_ABs) -- table back
    local label = torch.FloatTensor(bMap_Rs[1]:size()):fill(real_label)
    local labels = {}
    if opt.nGPU>0 then
    	label = label:cuda()
    end
    for i = 1, opt.nStack do
        table.insert(labels, label)
    end
    local errD_real = criterionDisc:forward(bMap_Rs, labels) -- sigma p*log(p_)  q*log(q_)  that is 1*log(real)  (1-0)log(1-p_) pre real how much loss + pre fake how much loss
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
    local errD_fake = criterionDisc:forward(bMap_Fs, labels)
    errD = (errD_real + errD_fake)/2

    -- errG
    local label = torch.FloatTensor(bMap_Fs[1]:size()):fill(real_label)
    local labels = {} -- true table
    if opt.nGPU>0 then
        label = label:cuda()
    end
    for i = 1, opt.nStack do
        table.insert(labels, label)
    end
    errG = criterionDisc:forward(bMap_Fs, labels)

    -- errL1
    local target
    if(opt.nStack > 1) then
        target = {}
        -- Same ground truth for all 8 stacks
        for st = 1, opt.nStack do
            table.insert(target, real_B)
        end
    else
        target = real_B
    end
    errL1 = criterionAE:forward(fake_Bs, target)  -- nStack tables diff error
    errAll = errG + errD + errL1 * opt.lambda -- weighted the weight

    -- image save out
    local idStg = opt.nStack  -- which stage to save out
    if opt.evaluate then    -- only save when evaluate  opt.evaluate
        if batchNumber <= opt.numOutImgs  then
            -- save image to folder only last stage at this time
            for i= 1,opt.batchSize do
                jMap_sum = jMaps[i]:clone():sum(1):squeeze()   -- 2 dim image
                image.save(paths.concat(opt.outImgsDir, 'test' .. '_sq' .. batchNumber - opt.batchSize + i .. '_st' .. idStg .. '_O.'.. opt.outFormat), deNormImg(fake_Bs[opt.nStack][i]:clone())) -- last output image [nstacks][batchSize]
                if opt.ifAllStgs == 1 then
                    --local j
                    for j = 1, opt.nStack -1 do
                        image.save(paths.concat(opt.outImgsDir, 'test' .. '_sq' .. batchNumber - opt.batchSize + i .. '_st' .. j .. '_O.'.. opt.outFormat), deNormImg(fake_Bs[j][i]:clone()))
                    end
                end
                image.save(paths.concat(opt.outImgsDir, 'test' .. '_sq' .. batchNumber - opt.batchSize + i .. '_st' .. idStg .. '_A.'.. opt.outFormat), deNormImg(inputs[i]:clone()))
                image.save(paths.concat(opt.outImgsDir, 'test' .. '_sq' .. batchNumber - opt.batchSize + i ..'_st' .. idStg ..  '_B.'.. opt.outFormat), deNormImg(real_B[i]:clone()))    -- deNormImg will change the value itself
                image.save(paths.concat(opt.outImgsDir, 'test' .. '_sq' .. batchNumber - opt.batchSize + i ..'_st' .. idStg ..  '_S.'.. opt.outFormat), jMap_sum)    -- deNormImg will change the value itself
                if opt.ifABO==1 then
                    --print('j is', j)
                     image.save(paths.concat(opt.outImgsDir, 'test' .. '_sq' .. batchNumber - opt.batchSize + i .. '_st' .. idStg .. '_ABO.'.. opt.outFormat), torch.cat({deNormImg(inputs[i]:clone()), deNormImg(real_B[i]:clone()), deNormImg(fake_Bs[opt.nStack][i]:clone())}, 3))
                end
                if opt.ifASO ==1 then
                     image.save(paths.concat(opt.outImgsDir, 'test' .. '_sq' .. batchNumber - opt.batchSize + i .. '_st' .. idStg .. '_ABS.'.. opt.outFormat), torch.cat({deNormImg(inputs[i]:clone()), jMap_sum, deNormImg(fake_Bs[opt.nStack][i]:clone())}, 3))
                end
            end
            -- display result
        end
    end

    cutorch.synchronize()
    lossG = lossG + errG
    lossD = lossD + errD
    lossL1 = lossL1 + errL1
    lossAll = lossAll + errAll

    if(opt.evaluate) then   -- str for other standard, no suitable here
        --print(string.format('Testing [%d/%d] \t Loss %.8f \t', batchNumber, nTest, err))
        print(string.format('Testing [%d/%d] \t errG %.8f \t errD %.8f \t errL1 %.8f \t errAll %.8f \t', batchNumber, nTest, errG, errD, errL1, errAll))
    else
        --print(string.format('Epoch: Testing [%d][%d/%d] \t Loss %.8f \t', epoch, batchNumber, nTest, err))
        print(string.format('Testing [%d][%d/%d] \t errG %.8f \t errD %.8f \t errL1 %.8f \t errAll %.8f \t', epoch, batchNumber, nTest, errG, errD, errL1, errAll))
    end
end
