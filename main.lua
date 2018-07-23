require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'
require 'nnx'
require 'nngraph'
matio = require 'matio'
-- display already in train
paths.dofile('TrainPlotter.lua')
torch.setdefaulttensortype('torch.FloatTensor')

local opts = paths.dofile('opts.lua')
opt = opts.parse(arg)

local heatmapSize = opt.inSize[2]/4 --64

if(opt.upsample) then
    heatmapSize = 256
end

--opt.data = paths.concat(opt.dataRoot, 'SURREAL/data', opt.datasetname)
opt.data = paths.concat(opt.dataRoot,opt.datasetname)   -- directly to cmu folder
local logDir = opt.logRoot .. '/' .. opt.datasetname -- cnn_saves/cmu/ change to ~/exp/GPM/cmu

opt.save     = paths.concat(logDir, opt.dirName) -- ../cmu/GPM7  set in sh, change to  ~/exp/GPM now
opt.cache    = paths.concat(logDir, 'cache')   -- cmu/cache
--os.execute('rm -f ' .. tmpfile)
-- clean previous plot.json

opt.plotter  = TrainPlotter.new(paths.concat(opt.save, 'plot.json'))  --cmu/plot.json
opt.outImgsDir = paths.concat(opt.save, 'outImgs')        -- /cmu/GPM/outImgs during test
opt.outGenFd = paths.concat(opt.save, 'outGenFd')        -- fd name added in main
opt.outGenImFd = paths.concat(opt.save, 'outGenImFd')
opt.outGenVid = paths.concat(opt.save,'outGenVid')  -- put parent paths
print('outGenImFd is', opt.outGenImFd)
--os.execute('mkdir -p ' .. opt.logRoot)
os.execute('mkdir -p ' .. opt.save) -- lua library os
os.execute('mkdir -p ' .. opt.outImgsDir)
os.execute('mkdir -p ' .. opt.cache)
os.execute('mkdir -p ' .. opt.outGenFd)
os.execute('mkdir -p ' .. opt.outGenImFd)
os.execute('mkdir -p ' .. opt.outGenVid)
os.execute('mkdir -p ' .. opt.dsGenFd)


-- Dependent on the ground truth
opt.outSize = {opt.inSize[2],opt.inSize[3]} -- ch h, w


-- Continue stopped training
if(opt.continue) then --epochNumber continue from there
    print('Continuing from epoch ' .. opt.epochNumber)  -- continue from where
    opt.retrainG = opt.save .. '/netG_' .. opt.epochNumber -1 ..'.t7' -- overwrites opt.retrain
    opt.retrainD = opt.save .. '/netD_' .. opt.epochNumber -1 .. '.t7'
    opt.optimStateG = opt.save .. '/optimStateG_'.. opt.epochNumber -1  ..'.t7'
    if opt.cGAN then
        opt.optimStateD = opt.save .. '/optimStateD_'.. opt.epochNumber -1  ..'.t7'
    end
    local backupDir = opt.save .. '/delete' .. os.time()
    os.execute('mkdir -p ' .. backupDir)
    os.execute('cp ' .. opt.save .. '/train.log ' ..backupDir)
    os.execute('cp ' .. opt.save .. '/test.log ' ..backupDir)
    os.execute('cp ' .. opt.save .. '/plot.json ' ..backupDir)
end

if opt.epochNumber ~= 1 then -- 1 from scratch
     print('Continuing from epoch ' .. opt.epochNumber)  -- continue from where
    opt.retrainG = opt.save .. '/netG_' .. opt.epochNumber -1 ..'.t7' -- overwrites opt.retrain
    opt.retrainD = opt.save .. '/netD_' .. opt.epochNumber -1 .. '.t7'
    opt.optimStateG = opt.save .. '/optimStateG_'.. opt.epochNumber -1  ..'.t7'
    opt.optimStateD = opt.save .. '/optimStateD_'.. opt.epochNumber -1  ..'.t7'
end

print(opt)  -- initial output opt params 
torch.save(paths.concat(opt.save, 'opt' .. os.time() .. '.t7'), opt)    -- save options
-- save to segmscratch,
cutorch.setDevice(opt.GPU) -- by default, use GPU 1
torch.manualSeed(opt.manualSeed)
print('LR ' .. opt.LR)      -- learning rate 
print('Saving everything to: ' .. opt.save)

paths.dofile('util.lua')    -- load lib, getCenter, getDataParallel,
paths.dofile('model.lua')   -- create or load model
paths.dofile('data.lua')    -- threads donkeys, trainLoadder in thread, hook functions
paths.dofile('train.lua')   -- optimState Logger, pram grad pram, trainBatch, update net; train(), add job
paths.dofile('test.lua')
--paths.dofile('testGenFd.lua')
paths.dofile('testGenFd1.lua')  -- choose the test folder files
--paths.dofile('eval.lua')  -- no need then

lossLi_tr = {} -- global to hold error li
lossLi_tst = {}
idWinPlt = 10
if(opt.evaluate) then   -- default false
    print('running evaluate')
    test()
    --matio.save(paths.concat(opt.save, 'lossesTst.Mat'), {lossLi_tst = lossLi_tst})
elseif opt.flgGenFd then
    print('generate reposed images from', opt.genFd)
    testGenFd()
    --testGenFd1()
elseif opt.flgGenIm then
    print('generate image with multiple poses')
    testGenIm() -- generate target images into outGenImFd
elseif opt.flgGenVid then
    print('generating video sequences')
    testGenVideo()
elseif opt.ifGenDs then
    print('generating dataset from', paths.concat(opt.dsSrcFd,opt.dsSrcNm))
    if string.find(opt.dsSrcNm, 'SURREAL') then
        testGenDs_SURREAL()
    else
        testGenDs()
    end
elseif opt.ifTsRMS then
    print('compare and calculate RMS of reposing images')
    testRMS()
else
    print('running train')
    print('clear previous json')
    if opt.epochNumber==1 then -- only remove if the training session is greater than 1
        print('clean previous plot.json file')
        os.execute('rm ' .. paths.concat(opt.save,  'plot.json'))   -- redraw json file
    end

    epoch = opt.epochNumber     -- epoch from 1 ? 
    for i=opt.epochNumber,opt.nEpochs do   -- 30 epoches, each 2000 images    each epoch, train one loop
        train()     -- get into the trin functions
        test()      -- save a few image out no test now, every epoch save out

        -- update server images
        local x = torch.linspace(1,#lossLi_tr,#lossLi_tr)
        print(string.format("epoch [%d] test loss list of train and test",epoch))
        print('train loss', lossLi_tr)
        print('test loss', lossLi_tst)

        if 0 == i%10 or i == opt.nEpochs then -- save less frequently
            saveDataParallel(paths.concat(opt.save, 'netG_' .. epoch .. '.t7'), netG)
            torch.save(paths.concat(opt.save, 'optimStateG_' .. epoch .. '.t7'), optimStateG)
            if opt.cGAN then
                saveDataParallel(paths.concat(opt.save, 'netD_' .. epoch .. '.t7'), netD)
                torch.save(paths.concat(opt.save, 'optimStateD_' .. epoch .. '.t7'), optimStateD)
            end
        end
        epoch = epoch + 1   -- update at last
    end
    --matio.save(paths.concat(opt.save, 'lossesTrTst.Mat'), {lossLi_tr = lossLi_tr, lossLi_tst = lossLi_tst})
    -- train only save
    matio.save(paths.concat(opt.save, 'lossTr.Mat'), {lossLi_tr = torch.Tensor(lossLi_tr)})
end
