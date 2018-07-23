require 'image'
require 'nn'

local timer = torch.Timer()
local loader = paths.dofile('loader_GPM.lua')
local dir = require 'pl.dir'


function testGenFd()
    local idxFrm = opt.idxPose     -- always use the 60 frames
    -- set to evaluate mode
    netG:evaluate()

    -- list all images
    local pthsImg = dir.getfiles(opt.genFd)
    local pthsVali = dir.getallfiles(paths.concat(opt.data, opt.testDir), '*_idxVali.mat')
    -- image number  imNum
    if #pthsVali< 2 then
        print('too little validation data')
        return
    end
    for i=1,#pthsImg do
        -- image in, joints2D in
        --local bsNm = paths.basename(pthsImg[i])
        local img = image.load(pthsImg[i], 3, 'float')
        local pthVali = pthsVali[i]
        local pth_mp4 = pthVali:sub(1,-13) .. '.mp4'
        local x_st, x_ed, idxVali = loader.getValiInfo(pth_mp4)
           -- joints2D in and process nothing to do with img now
          print('load from ', pth_mp4 )
        local joints2D = loader.loadJoints2D(pth_mp4, idxVali[math.min(idxFrm, idxVali:nElement())][1])

        local ratio = opt.inSize[2]/opt.loadSize[2]
        joints2D:select(2,1):csub(x_st[1][1]-1 )
        joints2D = joints2D * ratio
        local imgSc = cenCut(img, opt.inSize[2])  -- scaled before norm
        -- save out for reference
        -- jMap
        local jMap
       if 'limb' == opt.paraMap then
           jMap = loader.genMapLimbs2D(joints2D,opt.inSize[2], opt.inSize[3])
       elseif 'joint' == opt.paraMap then
           jMap = loader.genMapJoints2D(joints2D, opt.inSize[2], opt.inSize[3])
       else
            error('unknown param map type!')
       end
        imgOri = imgSc:clone()
        -- normalization
        img = normImg(imgSc)

        -- genMapSum
        local jMap_sum = jMap:sum(1):squeeze()

        local jMaps = nn.utils.addSingletonDimension(jMap,1)
        local imgs = nn.utils.addSingletonDimension(img,1)

        local outputs = netG:forward({ imgs, jMaps})

        local idStg = opt.nStack  -- which stage to save out
        local output = outputs[idStg][1]   -- only one batch

        -- denorm
        output = deNormImg(output)

        --save
        image.save(paths.concat(opt.outGenFd, 'gen' .. i .. '_st' .. idStg .. '_A.'.. opt.outFormat), imgOri)
        image.save(paths.concat(opt.outGenFd, 'gen' .. i .. '_st' .. idStg .. '_S.'.. opt.outFormat), jMap_sum) -- s for skeleton
        image.save(paths.concat(opt.outGenFd, 'gen' .. i .. '_st' .. idStg .. '_O.'.. opt.outFormat), output)
    end

end -- of test()
-----------------------------------------------------------------------------

