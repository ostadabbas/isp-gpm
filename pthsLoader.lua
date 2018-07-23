require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
local ffi = require 'ffi'
local class = require('pl.class')
local dir = require 'pl.dir'
local tablex = require 'pl.tablex'
local argcheck = require 'argcheck'
require 'sys'
require 'xlua'
require 'image'

local pthsLoader = torch.class('pthsLoader')
-- hold all img paths with shuffled order.  Train, test
local initcheck = argcheck{ 
   pack=true,
   help=[[
     A dataset class for images in a flat folder structure (folder-name is class-name).
     Optimized for extremely large datasets (upwards of 14 million images).
     Tested only on Linux (as it uses command-line linux utilities to scale up)
     History:
     1. Modified with pattern based searching, which I believe is more flexible. I change class name to pthsLoader which I found more proper for this class's purpose. This is GPM version
     Though, i hopes to combine everything together as one class. But I don't think that's a good design. All sample process actually needs to be customized. So working as a random path generator should be a best way.
     modified by, Shuangjun Liu (NEU/AClab),  1.26.2018
     -- imagePath       all image pths
     2. Add randomization control. get(i1,i2) function can always return same sequence which is helpful in test session to compare result.
]],

   {check=function(paths)
       local out = true;
       for k,v in ipairs(paths) do
          if type(v) ~= 'string' then
             print('paths can only be of string input');
             out = false
          end
       end
       return out
   end,
    name="paths", -- dataset/cmu/train
    type="table",
    help="Multiple paths of directories with images"},

   {name="split",
    type="number",
    help="Percentage of split to go to Training"
   },

   {name="samplingMode",
    type="string",
    help="Sampling mode: random | balanced ",
    default = "balanced"},

   {name="verbose",
    type="boolean",
    help="Verbose mode during initialization",
    default = false},

   {name="forceClasses",
    type="table",
    help="If you want this loader to map certain classes to certain indices, "
       .. "pass a classes table that has {classname : classindex} pairs."
       .. " For example: {3 : 'dog', 5 : 'cat'}"
       .. "This function is very useful when you want two loaders to have the same "
    .. "class indices (trainLoader/testLoader for example)",
    opt = true},

   {name="sampleHookTrain",
    type="function",
    help="applied to sample during training(ex: for lighting jitter). "
       .. "It takes the image path as input",
    opt = true},

   {name="sampleHookTest",
    type="function",
    help="applied to sample during testing",
    opt = true},

   {name="ifRand",
    type="boolean",
    help="if randomize the imgPath list",
    default = true},
}

function pthsLoader:__init(...)

   -- argcheck
   local args =  initcheck(...)
   print(args)
   for k,v in pairs(args) do self[k] = v end -- fil all fields

   -- find class names
   self.classes = {}  -- run0 run1 run2 
   local classPaths = {}
   if self.forceClasses then  -- nil 
      for k,v in pairs(self.forceClasses) do
         self.classes[k] = v
         classPaths[k] = {}
      end
   end
   local function tableFind(t, o) for k,v in pairs(t) do if v == o then return k end end end
   -- loop over each paths folder, get list of unique class names,
   -- also store the directory paths per class
   -- for each class,
   for k,path in ipairs(self.paths) do -- paths = {<fullpath>/train/}
      local dirs = dir.getdirectories(path);    -- train test and valid I think
      for k,dirpath in ipairs(dirs) do -- dirs = {01_01, 01_02, ...}
         local class = paths.basename(dirpath)
         local idx = tableFind(self.classes, class)
         if not idx then
            table.insert(self.classes, class) -- class = 'run0', 'run1', 'run2'
            idx = #self.classes  -- so path go to las one 
            classPaths[idx] = {} -- fill classes
         end
         if not tableFind(classPaths[idx], dirpath) then
            table.insert(classPaths[idx], dirpath); -- fill class name
         end
      end
   end

   self.classIndices = {} -- classIndices = 'run0' -> 3, 'run1' -> 1, 'run2' -> 2
   for k,v in ipairs(self.classes) do
      self.classIndices[v] = k
   end

   -- define command-line tools, try your best to maintain OSX compatibility
   local wc = 'wc'
   local cut = 'cut'
   local find = 'find'
   if jit.os == 'OSX' then
      wc = 'gwc'
      cut = 'gcut'
      find = 'gfind'
   end
   ----------------------------------------------------------------------
   -- Options for the GNU find command
   --local extensionList = opt.extension -- {'h5'} --{'jpg', 'png','JPG','PNG','JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
   --local findOptions = ' -maxdepth 2 -type f -iname "*.' .. extensionList[1] .. '"'
   -- --local findOptions = ' -type f -iname "*.' .. extensionList[1] .. '"'
   --for i=2,#extensionList do -- if more than 1
   --   findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
   --end

   -- pattern oriented finding method, I think this is more flexible
   local dataPtns =opt.dataPtns
   local findOptions = ' -maxdepth 2 -type f -iname "'.. dataPtns[1]..'"'
   for i = 2,#dataPtns do
      findOptions = findOptions .. '-o -iname "' .. dataPtns[i] .. '"'
   end

   -- find the image path names
   self.imagePath = torch.CharTensor()  -- path to each image in dataset
   self.imageClass = torch.LongTensor() -- class index of each image (class index in self.classes)
   self.classList = {}                  -- index of imageList to each image of a particular class
   self.classListSample = self.classList -- the main list used when sampling data -- same as classList.    1 : LongTensor - size: 25754 --2 : LongTensor - size: 12684 --3 : LongTensor - size: 16563 going from 1 to classSize
   -- classListTrain[idCls] indicates the indices for training for this idC

   print('running "find" on each class directory, and concatenate all'
         .. ' those filenames into a single file containing all image paths for a given class')
   -- so, generates one file per class
   local classFindFiles = {}
   for i=1,#self.classes do
      classFindFiles[i] = os.tmpname()    -- each a temp file, save mp4 list file names
   end
   local combinedFindList = os.tmpname(); -- all file names

   local tmpfile = os.tmpname()
   local tmphandle = assert(io.open(tmpfile, 'w'))
   -- iterate over classes eg run0 
   for i, class in ipairs(self.classes) do
      -- iterate over classPaths  eg 01_01
      for j,path in ipairs(classPaths[i]) do
         local command = find .. ' "' .. path .. '" ' .. findOptions
            .. ' >>"' .. classFindFiles[i] .. '" \n'
         tmphandle:write(command) -- write to file, execute 
      end
   end
   io.close(tmphandle)
   os.execute('bash ' .. tmpfile)
   os.execute('rm -f ' .. tmpfile)

   print('now combine all the files to a single large file')
   local tmpfile = os.tmpname()
   local tmphandle = assert(io.open(tmpfile, 'w'))
   -- concat all finds to a single large file in the order of self.classes
   for i=1,#self.classes do
      local command = 'cat "' .. classFindFiles[i] .. '" >>' .. combinedFindList .. ' \n'
      tmphandle:write(command)
   end
   io.close(tmphandle)
   os.execute('bash ' .. tmpfile)
   os.execute('rm -f ' .. tmpfile)

   --==========================================================================
   print('load the large concatenated list of sample paths to self.imagePath')
   local maxPathLength = tonumber(sys.fexecute(wc .. " -L '"
                                                  .. combinedFindList .. "' |"
                                                  .. cut .. " -f1 -d' '")) + 1
   local length = tonumber(sys.fexecute(wc .. " -l '"
                                           .. combinedFindList .. "' |"
                                           .. cut .. " -f1 -d' '"))
   assert(length > 0, "Could not find any image file in the given input paths")
   assert(maxPathLength > 0, "paths of files are length 0?")
   self.imagePath:resize(length, maxPathLength):fill(0)  -- char tensor
   local s_data = self.imagePath:data() -- all combinedFindList stored here
   local count = 0      -- image is all file list in order
   for line in io.lines(combinedFindList) do
      ffi.copy(s_data, line)  -- copy(dst, source, len)
      s_data = s_data + maxPathLength
      if self.verbose and count % 10000 == 0 then
         xlua.progress(count, length)
      end;
      count = count + 1
   end

   self.numSamples = self.imagePath:size(1)
   if self.verbose then print(self.numSamples ..  ' samples found.') end
   --==========================================================================
   print('Updating classList and imageClass appropriately')
   self.imageClass:resize(self.numSamples)
   local runningIndex = 0
   for i=1,#self.classes do
      if self.verbose then xlua.progress(i, #(self.classes)) end
      local length = tonumber(sys.fexecute(wc .. " -l '"
                                              .. classFindFiles[i] .. "' |"
                                              .. cut .. " -f1 -d' '")) -- indicate a field with delimiter ' '
      -- progress show progress bar
      if length == 0 then
         error('Class has zero samples')
      else
         self.classList[i] = torch.linspace(runningIndex + 1, runningIndex + length, length):long() -- list of number 1 to N1,  next time n1+1 to n2,... 
         self.imageClass[{{runningIndex + 1, runningIndex + length}}]:fill(i)
      end
      runningIndex = runningIndex + length
   end

   --==========================================================================
   -- clean up temporary files
   print('Cleaning up temporary files')
   local tmpfilelistall = ''
   for i=1,#(classFindFiles) do
      tmpfilelistall = tmpfilelistall .. ' "' .. classFindFiles[i] .. '"'
      if i % 1000 == 0 then
         os.execute('rm -f ' .. tmpfilelistall)
         tmpfilelistall = ''
      end
   end  -- delete every 1000
   os.execute('rm -f '  .. tmpfilelistall)
   os.execute('rm -f "' .. combinedFindList .. '"')
   --==========================================================================

   if self.split == 100 then
      self.testIndicesSize = 0
   else
      print('Splitting training and test sets to a ratio of '
               .. self.split .. '/' .. (100-self.split))
      self.classListTrain = {}
      self.classListTest  = {}
      self.classListSample = self.classListTrain
      local totalTestSamples = 0
      -- split the classList into classListTrain and classListTest
      for i=1,#self.classes do
         local list = self.classList[i]
         local count = self.classList[i]:size(1)  -- this class count 
         local splitidx = math.floor((count * self.split / 100) + 0.5) -- +round
         local perm
          if self.ifRand == true then
              perm = torch.randperm(count) -- random permutations
          else
              perm = torch.linspace(1, count, count)    -- ordered sequence
          end
         --=
         self.classListTrain[i] = torch.LongTensor(splitidx)
         for j=1,splitidx do
            self.classListTrain[i][j] = list[perm[j]]
         end
         if splitidx == count then -- all samples were allocated to train set
            self.classListTest[i]  = torch.LongTensor()
         else
            self.classListTest[i]  = torch.LongTensor(count-splitidx)
            totalTestSamples = totalTestSamples + self.classListTest[i]:size(1)
            local idx = 1
            for j=splitidx+1,count do
               self.classListTest[i][idx] = list[perm[j]]
               idx = idx + 1
            end
         end
      end  -- classListTest[i][idx]  i, run0, idx 01_01
      -- Now combine classListTest into a single tensor
      self.testIndices = torch.LongTensor(totalTestSamples)
      self.testIndicesSize = totalTestSamples
      local tdata = self.testIndices:data()
      local tidx = 0
      for i=1,#self.classes do
         local list = self.classListTest[i]
         if list:dim() ~= 0 then
            local ldata = list:data()
            for j=0,list:size(1)-1 do
               tdata[tidx] = ldata[j]
               tidx = tidx + 1
            end
         end
      end
   end
end

-- size(), size(class)
function pthsLoader:size(class, list)
   list = list or self.classList
   if not class then
      return self.numSamples
   elseif type(class) == 'string' then
      return list[self.classIndices[class]]:size(1)
   elseif type(class) == 'number' then
      return list[class]:size(1)
   end
end

-- getByClass
function pthsLoader:getByClass(class)
   local classindex = math.ceil(torch.uniform() * self.classListSample[class]:nElement()) -- random take one 
   local index = self.classListSample[class][classindex] -- indices in this class random class, random img belonging to this class
   local imgpath = ffi.string(torch.data(self.imagePath[index]))  -- data to the first address, then  change
   -- check if the file is a mat, then change it back to media file
   if string.match(imgpath, '%.mat') then -- the _idxVali.mat is at -13 location
      imgpath = imgpath:sub(1,-13) .. '.mp4'
   end
   local input, label, jMap = self:sampleHookTrain(imgpath) -- hook functions defined in
   return input, label, jMap, index
end

function pthsLoader:getMediaNm(idx)
   local imgpath = ffi.string(torch.data(self.imagePath[idx]))
   if string.match(imgpath, '%.mat') then -- the _idxVali.mat is at -13 location
       imgpath = imgpath:sub(1,-13) .. '.mp4'
   end
   return imgpath
end

-- sampler, samples from the training set.
function pthsLoader:sample(quantity) -- quantity batch size
  assert(quantity)
  --print('in sample function now')
  local inputs, labels, indices, jMaps
  inputs = torch.Tensor(quantity, opt.inSize[1], opt.inSize[2], opt.inSize[3])
  labels = torch.Tensor(quantity, opt.inSize[1], opt.inSize[2], opt.inSize[3])  -- outSize 64x64
   jMaps = torch.Tensor(quantity, #opt.jointsIx, opt.inSize[2], opt.inSize[3])
  indices = torch.Tensor(quantity) -- quantity 16, 128, 128
  for i=1,quantity do
      local class = torch.random(1, #self.classes)
      -- print('in batch number right now:', i)
      local input, label, jMap, index = self:getByClass(class)  -- random pick one
      -- print('sample read in at image', i)
      while (input == nil) do -- if the sample is nil for some reason (not read properly, size mismatch etc.) sample another one
         input, label, jMap, index = self:getByClass(class) -- !!
      end
     --print('in func pthsLoader before copy' )
     --print('input size', input:size())
     --print('label size', label:size())
     --print('jMap size', jMap:size())

      inputs[i]:copy(input)
      labels[i]:copy(label)
      jMaps[i]:copy(jMap)
      indices[i] = index
   end
     --print('in func pthsLoader after copy, batch ts' )
     --print('input size', inputs:size())
     --print('label size', labels:size())
     --print('jMap size', jMaps:size())
   return inputs, labels, jMaps, indices
end

function pthsLoader:get(i1, i2) -- get data from where to where
   local indices = torch.range(i1, i2);
   local quantity = i2 - i1 + 1;
   assert(quantity > 0)
   -- now that indices has been initialized, get the samples
  local inputs, labels, jMaps
   inputs = torch.Tensor(quantity, opt.inSize[1], opt.inSize[2], opt.inSize[3])
    labels = torch.Tensor(quantity, opt.inSize[1], opt.inSize[2], opt.inSize[3])  -- outSize 64x64
   jMaps = torch.Tensor(quantity, #opt.jointsIx, opt.inSize[2], opt.inSize[3])

   for i=1,quantity do
      -- load the sample
      local imgpath = ffi.string(torch.data(self.imagePath[indices[i]]))
      -- make to mp4 file
      if string.match(imgpath, '%.mat') then -- the _idxVali.mat is at -13 location
          imgpath = imgpath:sub(1,-13) .. '.mp4'
       end
      local input, label, jMap = self:sampleHookTest(imgpath)
      if(input == nil) then
        input = torch.Tensor(opt.inSize[1], opt.inSize[2], opt.inSize[3]):zero():add(0.5)
        label =torch.Tensor(quantity, opt.inSize[1], opt.inSize[2], opt.inSize[3]):zero():add(1)
         jMap = torch.Tensor(quantity, #opt.jointsIx, opt.inSize[2], opt.inSize[3]):zero()
      end
      inputs[i]:copy(input)
      labels[i]:copy(label)
      jMaps[i]:copy(jMap)
   end
    --print('from get func')
    --print('i1 and i2 are: ', i1, i2)
    --print('the tensor size of inputs, labels and jMaps are', inputs:size(), labels:size(), jMaps:size())
   return inputs, labels, jMaps, indices
end

return pthsLoader
