require 'image'
local ffi=require 'ffi'
------ Some FFI stuff used to pass storages between threads ------------------
ffi.cdef[[
void THFloatStorage_free(THFloatStorage *self);
void THLongStorage_free(THLongStorage *self);
]]

function makeDataParallel(model, nGPU) -- clong\e to all gpus
   if nGPU > 1 then
      print('converting module to nn.DataParallelTable')
      assert(nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
      local model_single = model
      model = nn.DataParallelTable(1)
      for i=1, nGPU do
         cutorch.setDevice(i)
         model:add(model_single:clone():cuda(), i)  -- table insert
      end
      cutorch.setDevice(opt.GPU)
   end
   return model
end

local function cleanDPT(module)
    -- This assumes this DPT was created by the function above: all the
    -- module.modules are clones of the same network on different GPUs
    -- hence we only need to keep one when saving the model to the disk.
    local newDPT = nn.DataParallelTable(1)
    cutorch.setDevice(opt.GPU)
    newDPT:add(module:get(1), opt.GPU)
    return newDPT
end

function saveDataParallel(filename, model)
    if torch.type(model) == 'nn.DataParallelTable' then
        torch.save(filename, cleanDPT(model))
    elseif torch.type(model) == 'nn.Sequential' then
        local temp_model = nn.Sequential()
        for i, module in ipairs(model.modules) do
            if torch.type(module) == 'nn.DataParallelTable' then
                temp_model:add(cleanDPT(module))
            else
                temp_model:add(module)
            end
        end
        torch.save(filename, temp_model)
    else
        torch.save(filename, model)
        print('The saved model is not a Sequential or DataParallelTable module.')
    end
end

function loadDataParallel(filename, nGPU)
    -- build paralle, depend on the type, add
    local model = torch.load(filename)
    if torch.type(model) == 'nn.DataParallelTable' then
        return makeDataParallel(model:get(1):float(), nGPU)
    elseif torch.type(model) == 'nn.Sequential' then
        for i,module in ipairs(model.modules) do
            if torch.type(module) == 'nn.DataParallelTable' then
                model.modules[i] = makeDataParallel(module:get(1):float(), nGPU)
            end
        end
        return model
    else
        print('The loaded model is not a Sequential or DataParallelTable module.')
        return model
    end
end

function setFloatStorage(tensor, storage_p)
    assert(storage_p and storage_p ~= 0, "FloatStorage is NULL pointer");
    local cstorage = ffi.cast('THFloatStorage*', torch.pointer(tensor:storage()))
    if cstorage ~= nil then
        ffi.C['THFloatStorage_free'](cstorage)
    end
    local storage = ffi.cast('THFloatStorage*', storage_p)
    tensor:cdata().storage = storage
end

function setLongStorage(tensor, storage_p)
    assert(storage_p and storage_p ~= 0, "LongStorage is NULL pointer");
    local cstorage = ffi.cast('THLongStorage*', torch.pointer(tensor:storage()))
    if cstorage ~= nil then
       ffi.C['THLongStorage_free'](cstorage)
    end
    local storage = ffi.cast('THLongStorage*', storage_p)
    tensor:cdata().storage = storage
end

function sendTensor(inputs) -- get tensor infor
    local size = inputs:size()
    local ttype = inputs:type()
    local i_stg =  tonumber(ffi.cast('intptr_t', torch.pointer(inputs:storage())))
    inputs:cdata().storage = nil
    return {i_stg, size, ttype}
end

function receiveTensor(obj, buffer)
    local pointer = obj[1]
    local size = obj[2]
    local ttype = obj[3]
    if buffer then
        buffer:resize(size)
        assert(buffer:type() == ttype, 'Buffer is wrong type')
    else
        buffer = torch[ttype].new():resize(size)      
    end
    if ttype == 'torch.FloatTensor' then
        setFloatStorage(buffer, pointer)
    elseif ttype == 'torch.LongTensor' then
        setLongStorage(buffer, pointer)
    else
       error('Unknown type')
    end
    return buffer
end

function getDir(dirName)    -- list all dirs  in a folder
    dirs = paths.dir(dirName)
    table.sort(dirs, function (a,b) return a < b end)
    for i = #dirs, 1, -1 do
        if(dirs[i] == '.' or dirs[i] == '..') then
            table.remove(dirs, i)
        end
    end
    return dirs
end

function getTightBox(label)
    -- Tighest bounding box covering the joint positions
    local tBox = {}
    tBox.x_min = label[{{}, {1}}]:min()
    tBox.y_min = label[{{}, {2}}]:min()
    tBox.x_max = label[{{}, {1}}]:max()
    tBox.y_max = label[{{}, {2}}]:max()
    tBox.humWidth  = tBox.x_max - tBox.x_min + 1
    tBox.humHeight = tBox.y_max - tBox.y_min + 1

    -- Slightly larger area to cover the head/feet of the human
    tBox.x_min = tBox.x_min - 0.25*tBox.humWidth -- left
    tBox.y_min = tBox.y_min - 0.35*tBox.humHeight -- top
    tBox.x_max = tBox.x_max + 0.25*tBox.humWidth -- right
    tBox.y_max = tBox.y_max + 0.25*tBox.humHeight -- bottom
    tBox.humWidth  = tBox.x_max - tBox.x_min + 1
    tBox.humHeight = tBox.y_max - tBox.y_min +1

    return tBox
end

function getCenter(label)
    local tBox = getTightBox(label)
    local center_x = tBox.x_min + tBox.humWidth/2
    local center_y = tBox.y_min + tBox.humHeight/2

    return {center_x, center_y}
end

function getScale(label, imHeight)
    local tBox = getTightBox(label)
    return math.max(tBox.humHeight/240, tBox.humWidth/240)
end

function pause()
    io.stdin:read'*l'
end

function table2str ( v )
    if "string" == type( v ) then
        v = string.gsub( v, "\n", "\\n" )
        if string.match( string.gsub(v,"[^'\"]",""), '^"+$' ) then
          return "'" .. v .. "'"
        end
        return '"' .. string.gsub(v,'"', '\\"' ) .. '"'
    else
        return "table" == type( v ) and table.tostring( v ) or
          tostring( v )
    end
end

function table.key_to_str ( k )
    if "string" == type( k ) and string.match( k, "^[_%a][_%a%d]*$" ) then
        return k
    else
        return "[" .. table.val_to_str( k ) .. "]"
    end
end

function table.tostring( tbl )
    local result, done = {}, {}
    for k, v in ipairs( tbl ) do
        table.insert( result, table.val_to_str( v ) )
        done[ k ] = true
    end
    for k, v in pairs( tbl ) do
        if not done[ k ] then
            table.insert( result,
              table.key_to_str( k ) .. "=" .. table.val_to_str( v ) )
        end
    end
    return "{" .. table.concat( result, "," ) .. "}"
end

meanstd = torch.load(paths.thisfile('meanstd/meanRgb.t7'))
mean = meanstd.mean
std = meanstd.std

-- normalization and denormalization
function deNormImg(img)
    -- add mean and std to deNorm img
    for c = 1, #mean do
        if  std then img[{{c}, {}, {}}]:mul(std[c]) end
        if mean then img[{{c}, {}, {}}]:add(mean[c]) end
    end
    return img
end

function normImg(img)
    -- imgOut = img
    for c = 1, #mean do
        if mean then
            img[{{c}, {}, {}}]:add(-mean[c])
        end
        if  std then
            img[{{c}, {}, {}}]:div(std[c])
        end
    end
    return img
end

function cenCut(img,stdL)
    -- cut image from center as square and resize to stdL
    local ind_st, imCut
    local szDiff = img:size(2)- img:size(3) -- h - w
    if szDiff>0 then
        ind_st = math.ceil(szDiff/2)
        imCut = img:narrow(2, ind_st, img:size(3))
    elseif szDiff <0 then
        ind_st= math.ceil(-szDiff/2)
        imCut = img:narrow(3, ind_st, img:size(2))
    end
    imCut = image.scale(imCut, stdL, stdL)
    return imCut    -- return scaled cut
end

function sqrPadding(img,stdL)
    -- padded the image to stdL square image with 128 gray value
    -- history: 1 return the st and end index and also the padding form
    -- pad direction, 0 for no padding,  1 for vertical , 2 for horizontal padding, -1 for malfunction case
    local ind_st, ind_end
    local imgPadded = torch.Tensor(img:size(1), stdL, stdL):fill(0.5)
    local scaleTo
    local padDrct = -1
    --local szDiff = img:size(2)- img:size(3) -- h - w
    -- greater edge
    -- scale to
    -- fill the imgPadded
    local imgSc = image.scale(img, stdL)
    if img:size(2) > img:size(3) then
        padDrct = 2
        scaleTo = stdL/ img:size(2)
        ind_st = math.ceil((img:size(2) - img:size(3))/2 * scaleTo)
        --imgPadded:narrow(3, ind_st, imgScl:size(3)):set(imgScl)
        imgPadded[{{},{},{ind_st, ind_st+imgSc:size(3)-1}}] = imgSc
        ind_end = ind_st+ imgSc:size(3)-1
    elseif img:size(2) < img:size(3) then
        padDrct = 1
        scaleTo = stdL/ img:size(3)
        ind_st = math.ceil((img:size(3) - img:size(2))/2 * scaleTo)
        --imgPadded:narrow(2, ind_st, imgSc:size(2)):set(imgSc) -- narrow can't work
        imgPadded[{{}, {ind_st, ind_st+imgSc:size(2)-1}, {}}] = imgSc
        ind_end = ind_st + imgSc:size(2)-1
    else
        padDrct = 0
        imgPadded = imgSc
    end
    return imgPadded, ind_st, ind_end, padDrct     -- return scaled cut
end

function cropPadIm(img, ind_st, ind_end, padDrct)
    -- this is specially designed for cropping padded image to get rid of the gray margin
    -- based on the padDrct, crop the image from ind_st to ind_end,
    -- padDrct 1, for vertical, 2 for horizontal, 0 for no padding return original image
    local imgCropped
    if padDrct == 1 then
        --print('img size is', img:size())
        imgCropped = img[{{}, {ind_st, ind_end},{}}]
    elseif padDrct == 2 then
        imgCropped = img[{{}, {}, {ind_st, ind_end}}]
    elseif padDrct ==0 then
        imgCropped = img    -- original
    else
        error('the padDrct code is wrong')
    end
    --print('result cropped img has size', imgCropped:size())
    return  imgCropped
end
function getBB(joints_gt)
    -- get bounding box as (x_st, y_st, x_end, y_end)
    --print('the min joints_gt is ', joints_gt:min(1))
    local BB = joints_gt:min(1):cat(joints_gt:max(1),2)
    return BB:floor():long():squeeze()
end
function getRMSE(output, label, ifNonZero)
    -- calculate the RMSE of two images. ifNonZero set, then both zero area will be excluded. Using 2 BBs has overlap issues
    ifNonZero = ifNonZero or true
    local ix = torch.cmul(output:eq(0), label:eq(0))
    local nContex
    if ifNonZero then
        nContex = (1-ix):sum()  -- only both non zero elements
    else
        nContex = output:numel()
    end
    --print('nContex is', nContex)  number
    --print('output size', output:size())
    --print('label size', label:size())
    local diff = output - label
    --print('diff is', diff:size())
    local RMSE= diff:pow(2)
    RMSE = RMSE:sum()/nContex
    RMSE = math.sqrt(RMSE)
    return RMSE
end