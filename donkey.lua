--paths.dofile('dataset.lua') --local dataset = torch.class('dataLoader'),  all img paths
paths.dofile('pthsLoader.lua')
paths.dofile('util.lua')
paths.dofile('img.lua')
paths.dofile('eval.lua')

local loadSize = opt.loadSize
local inSize   = opt.inSize

require 'image'
matio = require 'matio'


-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
-- creat a trainLoader (dataLoader from dataset.lua) testLoarder 
-- define the Hook functions, actually, not loading data yet 
------------------------------------------

-- a cache file of the training metadata (if doesnt exist, will be created)
local trainCache = paths.concat(opt.cache, opt.trainDir .. 'Cache.t7')
local testCache = paths.concat(opt.cache, opt.testDir .. 'Cache.t7')   -- '../valCache.t7'
local meanstdCache = paths.concat(opt.cache, 'meanstdCache.t7')

-- Check for existence of opt.data
if not os.execute('cd ' .. opt.data) then
    error(("could not chdir to '%s'"):format(opt.data))
end

-- Merge body parts
function changeSegmIx(segm, s)
    -- remapping? 
    local out = torch.zeros(segm:size())
    for i = 1,#s do out[segm:eq(i)] = s[i] end
    return out
end

-- Loading functions
local loader = paths.dofile('loader_GPM.lua')   -- get infor from path

-- Mean/std for whitening
local meanstd = torch.load(paths.thisfile('meanstd/meanRgb.t7'))
local mean = meanstd.mean
local std = meanstd.std

-- function common to do processing on loaded image/label-----------------------------------
local Hook = function(self, path, set)
-- return rgb and label, with scale and rotations, each time read in one, actually, self is not used in this func, for class hook up only.
    --[[
    load in the t1 rgb, t2 label, jMap generated from t2 joints2D
    ]]
    collectgarbage()
    local rgbFull, rgb, label
    local jMap
    local joints2D

    local t1, t2    -- t1 as source t2 as target
    local iT = loader.getDuration(path)

    -- get _idxVali infor
    local x_st, x_ed, idxVali = loader.getValiInfo(path)  -- x_st 1x1 tensor


     if 'train' == set then
        -- wide separation version
        t1 = idxVali[math.random(1, math.ceil(idxVali:nElement()/2))][1]    -- first half,
        t2 = idxVali[math.random(math.ceil(idxVali:nElement()/2), idxVali:nElement())][1]   -- second half
    else    -- for test case always pick up first and 60 frames or lowest available
        t1 = idxVali[1][1]
        t2 = idxVali[math.min(60, idxVali:nElement())][1]
    end

    -- load input
    rgbFull = loader.loadRGB(path, t1)  -- original image, but norm to 1
    label = loader.loadRGB(path, t2)    -- yes, from same sequence

    --print('opt inSize is', opt.inSize[2])
    joints2D = loader.loadJoints2D(path, t2) -- [ 2 x nJoints]
    -- Check
    if (rgbFull == nil or joints2D == nil ) then
            if(opt.verbose) then print('no rgb or joints2D! ' .. path) end
            return nil, nil
    end
     -- crop
    rgb = rgbFull:narrow(3,x_st[1][1], opt.loadSize[2]) -- 240, from start position, keep the window
    label = label:narrow(3,x_st[1][1],opt.loadSize[2])
    local ratio = opt.inSize[2]/opt.loadSize[2]
    --print('ratio is', ratio)  -- 0.53
    joints2D:select(2,1):csub(x_st[1][1]-1 ) -- get rid of how far biased from first coord
    -- scale
    rgb = image.scale(rgb,opt.inSize[3], opt.inSize[2]) -- image use x,y target size
    label = image.scale(label, opt.inSize[3], opt.inSize[2])
    joints2D = joints2D * ratio


    -- Color augmentation
    if(set == 'train') then
        color_scls= torch.Tensor{torch.uniform(0.6,1.4), torch.uniform(0.6,1.4),torch.uniform(0.6,1.4)}
        for c=1, 3 do
            rgb[{{c}, {}, {}}]:mul(color_scls[c]):clamp(0,1)
            label[{{c},{}, {}}]:mul(color_scls[c]):clamp(0,1)
        end
    end
    -- normalization
    for c = 1, #mean do
        if mean then
            rgb[{{c}, {}, {}}]:add(-mean[c])
            label[{{c}, {}, {}}]:add(-mean[c])
        end
        if  std then
            rgb[{{c}, {}, {}}]:div(std[c])
            label[{{c}, {}, {}}]:div(std[c])
        end
    end
    if 'limb' == opt.paraMap then
        jMap = loader.genMapLimbs2D(joints2D,opt.inSize[2], opt.inSize[3])
    elseif 'joint' == opt.paraMap then
        jMap = loader.genMapJoints2D(joints2D, opt.inSize[2], opt.inSize[3])
    else
        error('unknown param map type!')
    end
    return rgb, label, jMap
end

--trainLoader & function to load the train image-----------------------------------
trainHook = function(self, path)
    -- print('trainHook called')
    return Hook(self, path, 'train')
end

if paths.filep(trainCache) then
    print('Loading train metadata from cache')
    trainLoader = torch.load(trainCache)    -- declare global in thread?!
    trainLoader.sampleHookTrain = trainHook
    --assert(trainLoader.paths[1] == paths.concat(opt.data, 'train'),
    --       'cached files dont have the same path as opt.data. Remove your cached files at: '
    --          .. trainCache .. ' and rerun the program')
else
    print('Creating train metadata')  -- dataLoader is a class defined in dataset.lua, it's easy to lose track this way. file name should be dataLoarder, which is better
    trainLoader = pthsLoader{
        paths = {paths.concat(opt.data, opt.trainDir)},
        split = 100,    -- 100 percent training
        verbose = true,
        forceClasses = opt.forceClasses
    } -- it uses the class name instead of variable dataset, random
    torch.save(trainCache, trainLoader) -- get file list, save to avoid these
    trainLoader.sampleHookTrain = trainHook
end
collectgarbage()

-- do some sanity checks on trainLoader
do
    local class = trainLoader.imageClass    --  no such field I am afraid
    local nClasses = #trainLoader.classes
    assert(class:max() <= nClasses, "class logic has error")
    assert(class:min() >= 1, "class logic has error")
end

--testLoader & function to load the test image-----------------------------------
testHook = function(self, path)
    return Hook(self, path, 'test')
end

if paths.filep(testCache) then
    print('Loading test metadata from cache')
    testLoader = torch.load(testCache)  -- path loader
    testLoader.sampleHookTest = testHook
    assert(testLoader.paths[1] == paths.concat(opt.data, opt.testDir),
        'cached files dont have the same path as opt.data. Remove your cached files at: '
        .. testCache .. ' and rerun the program')
else
    print('Creating test metadata')
    print('Test dir: ' .. opt.testDir)
    testLoader = pthsLoader{
        paths = {paths.concat(opt.data, opt.testDir)},
        split = 0,
        verbose = true,
        forceClasses = trainLoader.classes,
        ifRand = false
    }   -- test loader no rand
    torch.save(testCache, testLoader)
    testLoader.sampleHookTest = testHook
end
collectgarbage()
--------------------------------------------------------------------------------
