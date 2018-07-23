-- Load information for GPM purpose from SURREAL dataset. Generate LDPD for GPM.
-- Author: Shuangjun Liu (ACLab)  ( based on original work of SURREAL dataset)


cv = require 'cv'
require 'cv.videoio'
require 'cv.imgproc'
require 'torch'
matio = require 'matio'

local M = {}    -- package
-- a package name M.  
local function getMatFile(path, str)
    return paths.dirname(path) .. '/' .. paths.basename(path, 'mp4') .. str .. '.mat'
end

-- NUMBER OF FRAMES --
local function getDuration(path) -- can get it with OpenCV instead
    local zrot
    if pcall(function() zrot = matio.load( getMatFile(path, '_info'), 'zrot') end) then
        return zrot:nElement()
    else
        if(opt.verbose) then print('Zrot not loaded ' .. path) end
        return 0
    end
end

local function getValiInfo(path)
    bsNm = paths.basename(path, 'mp4')
    dirNm = paths.dirname(path)
    valiNm =paths.concat(dirNm, bsNm .. '_idxVali.mat')
    --print('valiNm is ', valiNm)     -- for test purpose only
    x_st = matio.load(valiNm, 'x_st')
    x_ed = matio.load(valiNm, 'x_ed')
    idxVali = matio.load(valiNm, 'idxVali')
    return x_st, x_ed, idxVali
end


local function loadCameraLoc(path)
    local camLoc
    if pcall(function() camLoc = matio.load( getMatFile(path, '_info'), 'camLoc') end) then
    else
        if(opt.verbose) then print('CamLoc not loaded ' .. path) end; return nil
    end
    if(camLoc == nil) then; return nil; end
    return camLoc[1]
end

-- RGB --
local function loadRGB(path, t)
    local cap = cv.VideoCapture{filename=path} -- cv in donkey 
    if nil == cap then -- get in 
        print('videoCap can not be created from ', path)
    end
    cap:set{propId=1, value=t-1} --CV_CAP_PROP_POS_FRAMES set frame number

    local rgb 
    if pcall(function() _, rgb = cap:read{}; rgb = rgb:permute(3, 1, 2):float()/255; -- 312 to c ,h , w from 240 320 3
        rgb = rgb:index(1, torch.LongTensor{3, 2, 1}) end) then -- 3,2,1 to
        return rgb      -- normalized 255 
    else
        if (opt.verbose) then print('Img not opened ' .. path,'at frame ', t-1) end
        return nil
    end
end



-- TS TO opencv img
local function ts2cvImg(img)
    if img:size(1) == 3 then -- if rgb
        img_cv = img:index(1, torch.LongTensor{3,2,1})
    end
    --print('image cv size', img_cv:size())
    img_cv = img_cv:permute(2,3,1)  -- channel back
    -- stg 1  just double back
    --img_cv = img_cv:double()
    --stg 2 multi 255 and change to byte
    img_cv = img_cv*255
    img_cv = img_cv:byte()
    return img_cv
end

-- opencv TO TS img
local function cv2tsImg(img)
    imgTs = img:permute(3, 1, 2):float()/255; -- 312 to c ,h , w from 240 320 3 -- h,w,c tim -> ts
    if imgTs:size(1) ==3 then -- if rgb
        imgTs = imgTs:index(1, torch.LongTensor{3, 2, 1}) -- r,g,b channel
    end
    return imgTs
end

-- JOINTS 2D MAP --
local function genMapJoints2D(joints2D, h, w)
    -- get in joints2D tensor generate ch x size maps
    -- ch = joints number
    ch = joints2D:size(1)
    joints2D = joints2D:ceil()
    jMap = torch.zeros(ch, h, w):zero()
    for i=1,joints2D:size(1) do
        jMap[{{},{joints2D[i][2]},{joints2D[i][1]}}]:fill(1) -- all ch to 1
    end
    return jMap
end

-- the parent list, I will give the root parent itself, so line reduces to a point
jtsP = {2,3,7,7,4,5,7,7,8,9,12,13,9,9,14,15}
-- Joints 2D MAP --
local function genMapLimbs2D(joints2D, h, w)
    -- draw line to parent if no parent, gives one dot
    ch = joints2D:size(1)
    joints2D = joints2D:ceil()
    mapLi = {}
    -- iter joints, draw joint to parent or itself, save to mapLi
    for i = 1,joints2D:size(1) do
        mapSlice = torch.ByteTensor(h,w,1):zero()
        cv.line{mapSlice, {joints2D[i][1],joints2D[i][2]} , {joints2D[jtsP[i]][1], joints2D[jtsP[i]][2]}, {255, 255, 255}}

        mapTs = cv2tsImg(mapSlice)
        mapLi[i] = mapTs -- still cv format
    end
    mapTs = torch.cat(mapLi,1)  -- first dim add together
    return  mapTs
end


-- JOINTS 2D --
local function loadJoints2D(path, t)
    --input:  t for frame ,path for mp4 basename
    local joints2D, vars
    if pcall(function() vars = matio.load( getMatFile(path, '_info'), 'joints2D') end) then
        if pcall(function ()
            -- body
            if (vars:dim()<3) then
                -- print(path .. ' joints2D only has dim '.. vars:dim())
                vars = nn.utils.addSingletonDimension(vars,3)
                -- print('modify the dimension to', vars:dim())
            end
            joints2D = vars[{{}, {}, { t }}]:squeeze():t():add(1); joints2D = joints2D:index(1, torch.LongTensor(opt.jointsIx))
        end) then 
        else print(path .. ' has weirdness (joints2D)' .. t); 
            print('vars read in is', vars)
            return nil end
        local zeroJoint2D = joints2D[{{}, {1}}]:eq(1):cmul(joints2D[{{}, {2}}]:eq(239)) -- Check if joints are all zeros.
        if zeroJoint2D:sum()/zeroJoint2D:nElement() == 1 then
            if(opt.verbose) then print('Skipping ' .. path .. '... (joints2D are all [0, 0])') end
            return nil
        end
    else
        if(opt.verbose) then print('Joints2D not loaded ' .. path) end
    end
    return joints2D
end

-- JOINTS 3D --
function loadJoints3D(path, t)
    local joints3D, vars
    if pcall(function() vars = matio.load( getMatFile(path, '_info'), 'joints3D') end) then
        if pcall(function() joints3D = vars[{{}, {}, { t }}]:squeeze():t(); joints3D = joints3D:index(1, torch.LongTensor(opt.jointsIx))  end) then       -- [24 x 3]
        else print(path .. ' has weirdness (joints3D)' .. t); return nil end
        local zeroJoint3D = joints3D[{{}, {1}}]:eq(0):cmul(joints3D[{{}, {2}}]:eq(0)):cmul(joints3D[{{}, {3}}]:eq(0)) -- Check if joints are all zeros.
        if zeroJoint3D:sum()/zeroJoint3D:nElement() == 1 then
            if(opt.verbose) then print('Skipping ' .. path .. '... (joints3D are all [0, 0])') end
            return nil
        end
    else
        if(opt.verbose) then print('Joints3D not loaded ' .. path) end
    end
    return joints3D
end

-- SEGMENTATION --
local function loadSegm(path, t)
    local segm
    if pcall(function() segm = matio.load( getMatFile(path, '_segm'), 'segm_' .. t) end) then -- [240 x 320]
    else
        if(opt.verbose) then print('Segm not loaded ' .. path) end;  return nil 
    end
    if(segm == nil) then; return nil; end
    segm = changeSegmIx(segm, {2, 12, 9, 2, 13, 10, 2, 14, 11, 2, 14, 11, 2, 2, 2, 1, 6, 3, 7, 4, 8, 5, 8, 5})  -- 24 seg
    return segm
end

-- DEPTH --
local function loadDepth(path, t, dPelvis)
    local depth, out, pelvis, mask, nForeground, lowB, upB
    if pcall(function() depth = matio.load( getMatFile(path, '_depth'), 'depth_' .. t) end) then -- [240 x 320]
    else
        if(opt.verbose) then print('Depth not loaded ' .. path) end;  return nil, nil
    end
    if(depth == nil) then; return nil, nil; end

    out = torch.zeros(depth:size())
    mask = torch.le(depth, 1e+3)  -- background =1.0000e+10
    nForeground = mask:view(-1):sum()  -- #foreground pixels
    lowB = -(opt.depthClasses - 1)/2 -- -9 lower bound 
    upB = (opt.depthClasses - 1)/2

    local fgix = torch.le(depth, 1e3)
    local bgix = torch.gt(depth, 1e3)
    out[fgix] = torch.cmax(torch.cmin(torch.ceil(torch.mul(torch.add(depth[fgix], -dPelvis), 1/opt.stp)), upB), lowB) -- align and quantize
    out[bgix] = lowB-1 -- background class -10 
    out = out:add(1+upB) -- so that it's between 0-19. It was [-10, -9, .. 0 .. 9].

    return out, nForeground 
end

M.getDuration   = getDuration
M.loadCameraLoc = loadCameraLoc
M.loadRGB       = loadRGB
M.loadJoints2D  = loadJoints2D
M.loadJoints3D  = loadJoints3D
M.loadSegm      = loadSegm
M.loadDepth     = loadDepth
M.ts2cvImg      = ts2cvImg
M.genMapJoints2D = genMapJoints2D
M.getValiInfo   = getValiInfo
M.jtsP          = jtsP
M.genMapLimbs2D = genMapLimbs2D

return M
