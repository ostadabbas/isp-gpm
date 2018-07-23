local ffi = require 'ffi'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

-- This script contains the logic to create K threads for parallel data-loading.
-- For the data-loading details, look at donkey.lua
-------------------------------------------------------------------------------
do -- start K datathreads (donkeys)
    if opt.nDonkeys > 0 then    -- 8 donkeys
        local options = opt -- make an upvalue to serialize over to donkey threads
        donkeys = Threads(
            opt.nDonkeys,
            function()
                require 'torch'
                paths.dofile('TrainPlotter.lua')
                require 'nn'
                cv = require 'cv'
                cv = require 'cv.imgproc'
                require 'cv.videoio'
            end,
            function(idx)
                opt = options -- pass to all donkeys via upvalue
                tid = idx
                local seed = opt.manualSeed + idx
                torch.manualSeed(seed)
                print(string.format('Starting donkey with id: %d seed: %d', tid, seed))
                paths.dofile('donkey.lua')  -- a hook func,  train_loader, loader is no upvalue but global what will happen 
            end
        ); -- initial func, each loader, save all indices 
    else -- single threaded data loading. useful for debugging
        paths.dofile('donkey.lua')
        donkeys = {}
        function donkeys:addjob(f1, f2) f2(f1()) end
        function donkeys:synchronize() end
    end
end

nClasses = nil
classes = nil
donkeys:addjob(function() return trainLoader.classes end, function(c) classes = c end)  -- end funcs, assign c to classes, but each thread will replace global classes this way, run0 run1 run2
donkeys:synchronize()
nClasses = #classes
assert(nClasses, "Failed to get nClasses")
print('nClasses: ', nClasses)
torch.save(paths.concat(opt.save, 'classes.t7'), classes)

nTest = 0   -- total test number
donkeys:addjob(function() return testLoader:size() end, function(c) nTest = c end)  -- all loader created, no actuall action yet 
donkeys:synchronize()
assert(nTest > 0, "Failed to get nTest")
print('nTest: ', nTest) -- to here nTest is in

