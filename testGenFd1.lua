--[[
generate images from the source folders GPM_genSource1, each folder contains multiple folders from different dataset.
List all sub folder, create corresponding sub in outDir
Save result to corresponding dataset
]]--
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
    local pthsImg = dir.getallfiles(paths.concat('.', opt.genFd),'*.jpg')  -- all files
    local pthsVali = dir.getallfiles(paths.concat(opt.data, opt.testDir), '*_idxVali.mat')  -- test joint infor for re position

    -- check and mkdir
    dsFds = dir.getdirectories(opt.genFd)
    for i,v in ipairs(dsFds) do
        print('directory is', v)
        os.execute('mkdir -p ' .. paths.concat(opt.outGenFd, paths.basename(v)))
    end
    -- image number  imNum
    if #pthsVali< #pthsImg then
        print('too little validation data, joint infor should be larger than test image')
        return
    end
    for i=1,#pthsImg do
        -- image in, joints2D in
        --local bsNm = paths.basename(pthsImg[i])
        local img = image.load(pthsImg[i], 3, 'float')
        local dsNm = paths.basename(paths.dirname(pthsImg[i]))  -- parent folder
        local pthVali = pthsVali[i]
        local pth_mp4 = pthVali:sub(1,-13) .. '.mp4'
        local x_st, x_ed, idxVali = loader.getValiInfo(pth_mp4)
           -- joints2D in and process nothing to do with img now
          print('load from ', pth_mp4 )
        local joints2D = loader.loadJoints2D(pth_mp4, idxVali[math.min(idxFrm, idxVali:nElement())][1]) -- idxVali 2D tensor

        local ratio = opt.inSize[2]/opt.loadSize[2]
        joints2D:select(2,1):csub(x_st[1][1]-1 )
        joints2D = joints2D * ratio
        --local imgSc = cenCut(img, opt.inSize[2])  -- scaled before norm
        local imgSc, ind_st, ind_end, padDrct = sqrPadding(img, opt.inSize[2])    -- 128 stdL  imgSc original format float
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
        local imgOri = imgSc:clone()    -- source, ori padded version no normed
        -- normalization
        img = normImg(imgSc:clone())    -- source corrupted here

        -- genMapSum
        local jMap_sum = jMap:sum(1):squeeze()
       jMap_sum = jMap_sum:repeatTensor(3,1,1)     -- to 3 channels

        local jMaps = nn.utils.addSingletonDimension(jMap,1)
        local imgs = nn.utils.addSingletonDimension(img,1)

        local outputs = netG:forward({ imgs, jMaps})

        local idStg = opt.nStack  -- which stage to save out
        local output = outputs[idStg][1]   -- only one batch
        local outputsCropped = {}
        local imgSv = img -- original size version

        -- denorm
        output = deNormImg(output)

        -- cropping to original size
        if opt.ifCrop==1 then
            jMap_sum = cropPadIm(jMap_sum, ind_st, ind_end, padDrct)

            output = cropPadIm(output, ind_st, ind_end, padDrct)
            if opt.ifAllStgs then
                for j = 1, opt.nStack -1 do
                    outputsCropped[j] = cropPadIm(deNormImg(outputs[j][1]:clone()), ind_st, ind_end, padDrct)
                end
            end
        else
            for j = 1, opt.nStack -1 do
                imgSv = imgOri
                outputsCropped[j] = deNormImg(outputs[j][1]:clone())
            end
        end


        --save
        -- index version
        -- check size
        --print('imgOri size and type', imgOri:size(), imgOri:type())
        --print('output size and type', output:size(), output:type())
        --print('jMap_sum size and type', jMap_sum:size(), jMap_sum:type())
        image.save(paths.concat(opt.outGenFd, dsNm, 'gen' .. i .. '_st' .. idStg .. '_A.jpg'), imgOri)
        image.save(paths.concat(opt.outGenFd, dsNm, 'gen' .. i .. '_st' .. idStg .. '_S.jpg'), jMap_sum) -- s for skeleton
        image.save(paths.concat(opt.outGenFd, dsNm, 'gen' .. i .. '_st' .. idStg .. '_O.jpg'), output)
        if opt.ifASO==1 then
            --print('j is', j)
            print('gen ASO', i)
             image.save(paths.concat(opt.outGenFd, dsNm, 'gen' .. i .. '_st' .. idStg .. '_ASO.'.. opt.outFormat), torch.cat({ imgOri:clone(), jMap_sum:float(), output:float()}, 3))
        end
        if opt.ifAllStgs == 1 then
            for j = 1, opt.batchSize -1 do
                image.save(paths.concat(opt.outGenFd, dsNm, 'gen' .. i .. '_st' .. j .. '_O.jpg'), outputsCropped[j])
            end
        end

        -- file name version save
        --image.save(paths.concat(opt.srcGenDir, dsNm, pthsImg[i] .. '_st' .. idStg .. '_A.'.. opt.outFormat), imgOri)
        --image.save(paths.concat(opt.srcGenDir, dsNm, pthsImg[i] .. '_st' .. idStg .. '_S.'.. opt.outFormat), jMap_sum) -- s for skeleton
        --image.save(paths.concat(opt.srcGenDir, dsNm, pthsImg[i] .. '_st' .. idStg .. '_O.'.. opt.outFormat), output)

    end

end -- of test()

function testGenIm()        -- parameters are already in opt
    -- use all joint infor to augment one image into all poses
    local idxFrm = 60     -- always use the 60 frames
    -- set to evaluate mode
    netG:evaluate()
    -- list all images
    --local pthsImg = dir.getallfiles(opt.genFd)  -- all files
    local pthsVali = dir.getallfiles(paths.concat(opt.data, opt.testDir), '*_idxVali.mat')  -- test joint infor for re position

    -- max index within range of idxVali
    local idxValiFiles = torch.linspace(1,500, 500)

    if torch.Tensor(idxValiFiles):max() > #pthsVali then
        print('index exceeded the idxVali file number')
        return
    end
    -- iter through all idxVali and get joints2D infor
    -- generate images
    -- save to the target fd
    local imgOri = image.load(paths.concat(opt.genFd, opt.genImNm))
    local img
    local imgOriSave
    local imgSc, ind_st, ind_end, padDrct = sqrPadding(imgOri, opt.inSize[2])    -- 128 stdL
    imgOriSave = cropPadIm(imgSc, ind_st, ind_end, padDrct)

    if opt.ifCrop then
        image.save(paths.concat(opt.outGenImFd, 'ori_A.jpg'), imgOriSave)
    else
        image.save(paths.concat(opt.outGenImFd, 'ori_A.jpg'), imgSc)
    end

    for i=1,idxValiFiles:nElement() do
        -- image in, joints2D in
        --local bsNm = paths.basename(pthsImg[i])
        --local img = image.load(pthsImg[i], 3, 'float')
        --local dsNm = paths.basename(paths.dirname(pthsImg[i]))  -- parent folder
        local pthVali = pthsVali[idxValiFiles[i]]   -- specific index
        local pth_mp4 = pthVali:sub(1,-13) .. '.mp4'
        local x_st, x_ed, idxVali = loader.getValiInfo(pth_mp4)
           -- joints2D in and process nothing to do with img now
          print('load from ', pth_mp4 )
        local joints2D = loader.loadJoints2D(pth_mp4, idxVali[math.min(idxFrm, idxVali:nElement())][1]) -- idxVali 2D tensor

        local ratio = opt.inSize[2]/opt.loadSize[2]
        joints2D:select(2,1):csub(x_st[1][1]-1 )
        joints2D = joints2D * ratio
        --local imgSc = cenCut(img, opt.inSize[2])  -- scaled before norm
        --local imgSc, ind_st, ind_end, padDrct = sqrPadding(imgOri, opt.inSize[2])    -- 128 stdL

        -- jMap
        local jMap
       if 'limb' == opt.paraMap then
           jMap = loader.genMapLimbs2D(joints2D,opt.inSize[2], opt.inSize[3])
       elseif 'joint' == opt.paraMap then
           jMap = loader.genMapJoints2D(joints2D, opt.inSize[2], opt.inSize[3])
       else
            error('unknown param map type!')
       end
        --imgOri = imgSc:clone()
        -- normalization
        img = normImg(imgSc:clone())

        -- genMapSum
        local jMap_sum = jMap:sum(1):squeeze()
        --print('jmap_sum size is', jMap_sum:size())
        jMap_sum = jMap_sum:repeatTensor(3,1,1)     -- to 3 channels
        --print('after repeating jmap_sum size is', jMap_sum:size())
        local jMaps = nn.utils.addSingletonDimension(jMap,1)
        local imgs = nn.utils.addSingletonDimension(img,1)

        local outputs = netG:forward({ imgs, jMaps})    -- stack x batch x img

        local idStg = opt.nStack  -- which stage to save out
        local output = outputs[idStg][1]   -- only one batch
        local outputsCropped ={}
        -- denorm
        output = deNormImg(output)

        -- cropping to original size
        if opt.ifCrop then
            --imgSave = cropPadIm(imgOri, ind_st, ind_end, padDrct)

            jMap_sum = cropPadIm(jMap_sum, ind_st, ind_end, padDrct)

            output = cropPadIm(output, ind_st, ind_end, padDrct)
            if opt.ifAllStgs then
                for j = 1, opt.batchSize -1 do
                    outputsCropped[j] = cropPadIm(deNormImg(outputs[j][1]:clone()), ind_st, ind_end, padDrct)
                end
            end
        else
            for j = 1, opt.batchSize -1 do
                imgSave = imgOri
                outputsCropped[j] = deNormImg(outputs[j][1]:clone())
            end
        end

        --save
        -- index version
        --image.save(paths.concat(opt.outGenImFd, dsNm, 'gen' .. i .. '_st' .. idStg .. '_A.jpg'), imgOri)  -- no need to save original
        image.save(paths.concat(opt.outGenImFd,   'gen' .. i .. '_st' .. idStg .. '_S.jpg'), jMap_sum) -- s for skeleton    no dsNm here as we generate only 1 image per time
        image.save(paths.concat(opt.outGenImFd,   'gen' .. i .. '_st' .. idStg .. '_O.jpg'), output)
        if opt.ifAllStgs == 1 then
            for j = 1, opt.batchSize -1 do
                --output = deNormImg(outputs[j][1]:clone())
                image.save(paths.concat(opt.outGenImFd,  'gen' .. i .. '_st' .. j .. '_O.jpg'), outputsCropped[j])
            end
        end
        -- file name version save
        --image.save(paths.concat(opt.srcGenDir, dsNm, pthsImg[i] .. '_st' .. idStg .. '_A.'.. opt.outFormat), imgOri)
        --image.save(paths.concat(opt.srcGenDir, dsNm, pthsImg[i] .. '_st' .. idStg .. '_S.'.. opt.outFormat), jMap_sum) -- s for skeleton
        --image.save(paths.concat(opt.srcGenDir, dsNm, pthsImg[i] .. '_st' .. idStg .. '_O.'.. opt.outFormat), output)
    end
end -- of test()
function testGenVideo()
    -- generate videos of all files in genVid, according to sequence index of idxSeqGen
    local idxFrm = 60     -- always use the 60 frames
    -- set to evaluate mode
    netG:evaluate()

    -- list all images
    local pthsImg = dir.getallfiles(opt.genVid, '*.jpg')  -- all files
    local pthsVali = dir.getallfiles(paths.concat(opt.data, opt.testDir), '*_idxVali.mat')  -- test joint infor for re position

    for i=1,#pthsImg do
        -- image in, joints2D in
        --local bsNm = paths.basename(pthsImg[i])
        local img = image.load(pthsImg[i], 3, 'float')
        -- gen fd dir
        for j = 1, #opt.idxSeqGen   do  -- iter sequence
            local vidFd = paths.concat(opt.outGenVid, string.format(paths.basename(pthsImg[i],'jpg') .. '%04d', j))
            print('vidFd is', vidFd)
            os.execute('mkdir -p ' .. vidFd)   -- make file fd
            local imgFd  = paths.concat(vidFd,'imgs')
            print('img Fd is', imgFd)
            os.execute('mkdir -p ' .. imgFd)  -- imgs fd
            local pthVali = pthsVali[j]
            local pth_mp4 = pthVali:sub(1,-13) .. '.mp4'
            local x_st, x_ed, idxVali = loader.getValiInfo(pth_mp4)
              --print('load from ', pth_mp4 )
            -- build fds
            --string.format('Testing [%d/%d] \t Loss %.8f \t', batchNumber, nTest, err)

            for k = 1, idxVali:size(1) do  -- iter frames
                --print('total pthVali is ')
                local joints2D = loader.loadJoints2D(pth_mp4, idxVali[k][1]) -- idxVali 2D tensor
                local ratio = opt.inSize[2]/opt.loadSize[2]
                joints2D:select(2,1):csub(x_st[1][1]-1 )
                joints2D = joints2D * ratio
                --local imgSc = cenCut(img, opt.inSize[2])  -- scaled before norm
                local imgSc, ind_st, ind_end, padDrct = sqrPadding(img:clone(), opt.inSize[2])    -- 128 stdL  imgSc original format float
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
                --local imgOri = imgSc:clone()    -- source, ori padded version no normed
                -- normalization
                local imgIn = normImg(imgSc:clone())    -- source corrupted here

                -- genMapSum
                local jMap_sum = jMap:sum(1):squeeze()
               jMap_sum = jMap_sum:repeatTensor(3,1,1)     -- to 3 channels

                local jMaps = nn.utils.addSingletonDimension(jMap,1)
                local imgs = nn.utils.addSingletonDimension(imgIn,1)

                local outputs = netG:forward({ imgs, jMaps})

                local idStg = opt.nStack  -- which stage to save out
                local output = outputs[idStg][1]   -- last stage ,only one batch,
                local outputsCropped = {}
                -- denorm
                output = deNormImg(output)

                --save
                -- index version
                -- check size
                --print('imgOri size and type', imgOri:size(), imgOri:type())
                --print('output size and type', output:size(), output:type())
                --print('jMap_sum size and type', jMap_sum:size(), jMap_sum:type())
                image.save(paths.concat(imgFd, string.format('img%04d.jpg',k)), output) -- save only output images
            end
            -- gen video
            local cmd_ffmpeg = string.format('ffmpeg -y -r 30 -i "%s" -c:v h264 -pix_fmt yuv420p -crf 23 "%s.mp4"', paths.concat(imgFd,
            'img%04d.jpg'), paths.concat(vidFd,paths.basename(vidFd)))
            os.execute(cmd_ffmpeg)

        end
    end
end -- of test()

function testGenDs()
    local idxFrm = opt.idxPose     -- always use the 60 frames
    -- set to evaluate mode
    netG:evaluate()

    -- list all images
    local idxMPIts= torch.LongTensor({1,2,3,4,5,6,7,9,10,10,11,12,13,14,15,16})  -- idx to MPI from SURREAL 16, 10 head needs to be recalculated
    local dsSrcFd = paths.concat(opt.dsSrcFd, opt.dsSrcNm)  -- dsSrcNm manually collected, must fix
    local pthsImg = dir.getallfiles(dsSrcFd,'*.jpg')  -- all files
    local pthsVali = dir.getallfiles(paths.concat(opt.data, opt.testDir), '*_idxVali.mat')  -- test joint infor for re position

    -- check and mkdir
    --dsFds = dir.getdirectories(opt.genFd)
    --for i,v in ipairs(dsFds) do
    --    print('directory is', v)
    --    os.execute('mkdir -p ' .. paths.concat(opt.srcGenDir, paths.basename(v)))
    --end
    os.execute('mkdir -p ' .. opt.dsGenFd)  -- parent Fd
    os.execute('mkdir -p ' .. paths.concat(opt.dsGenFd, opt.dirName .. '_' .. opt.dsSrcNm))
    local dsFdSpec = paths.concat(opt.dsGenFd, opt.dirName .. '_' .. opt.dsSrcNm)
    local imGenFd = paths.concat(dsFdSpec, 'images')
    os.execute('mkdir -p ' .. imGenFd)  -- images fd
    print('generate dataset at ', dsFdSpec)
    -- image number  imNum

    if opt.nImgGenDs > #pthsVali or opt.nImgGenDs> #pthsImg then
        print('generated dataset can not exceed the number of images or valid file number')
        return
    end
    local joints_gt = torch.zeros(opt.nImgGenDs, 16,3)
    for i=1, opt.nImgGenDs do
        -- image in, joints2D in
        --local bsNm = paths.basename(pthsImg[i])
        local img = image.load(pthsImg[i], 3, 'float')
        local dsNm = paths.basename(paths.dirname(pthsImg[i]))  -- parent folder
        local pthVali = pthsVali[i]
        local pth_mp4 = pthVali:sub(1,-13) .. '.mp4'
        local x_st, x_ed, idxVali = loader.getValiInfo(pth_mp4)
           -- joints2D in and process nothing to do with img now
          print('load from ', pth_mp4 )
        local joints2D = loader.loadJoints2D(pth_mp4, idxVali[math.min(idxFrm, idxVali:nElement())][1]) -- idxVali 2D tensor

        local ratio = opt.inSize[2]/opt.loadSize[2]
        joints2D:select(2,1):csub(x_st[1][1]-1 )
        joints2D = joints2D * ratio
        local joints_MPI = joints2D:index(1,idxMPIts):clone()
        joints_MPI[10]= joints_MPI[9] + 3*(joints_MPI[9] - joints_MPI[8])
        joints_gt[i]:narrow(2,1,2):copy(joints_MPI) -- leave last 0

        --local imgSc = cenCut(img, opt.inSize[2])  -- scaled before norm
        local imgSc, ind_st, ind_end, padDrct = sqrPadding(img, opt.inSize[2])    -- 128 stdL  imgSc original format float
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
        local imgOri = imgSc:clone()    -- source, ori padded version no normed
        -- normalization
        img = normImg(imgSc:clone())    -- source corrupted here

        -- genMapSum
        local jMap_sum = jMap:sum(1):squeeze()
       jMap_sum = jMap_sum:repeatTensor(3,1,1)     -- to 3 channels

        local jMaps = nn.utils.addSingletonDimension(jMap,1)
        local imgs = nn.utils.addSingletonDimension(img,1)

        local outputs = netG:forward({ imgs, jMaps})

        local idStg = opt.nStack  -- which stage to save out
        local output = outputs[idStg][1]   -- only one batch

        -- denorm
        output = deNormImg(output)

        image.save(paths.concat(imGenFd, string.format('image_%06d.jpg', i)), output)
    end
    joints_gt = joints_gt:permute(3,2,1)
    matio.save(paths.concat(dsFdSpec, 'joints_gt.mat'), {joints_gt = joints_gt})
end -- of test()

function testGenDs_SURREAL()
    local idxFrm = opt.idxPose     -- always use the 60 frames
    -- set to evaluate mode
    netG:evaluate()

    -- list all images
    local idxMPIts= torch.LongTensor({1,2,3,4,5,6,7,9,10,10,11,12,13,14,15,16})  -- idx to MPI from SURREAL 16, 10 head needs to be recalculated
    --local dsSrcFd = paths.concat(opt.dsSrcFd, string.format(opt.dsSrcNm .. '_%d', opt.nImgGenDs))
    --local pthsImg = dir.getallfiles(dsSrcFd,'*.jpg')  -- all files
    local dsSrcNm = string.format(opt.dsSrcNm .. '_%d', opt.nImgGenDs)  -- SURREAL_100
    local pthsVali = dir.getallfiles(paths.concat(opt.data, opt.testDir), '*_idxVali.mat')  -- test joint infor for re position

    -- check and mkdir
    --dsFds = dir.getdirectories(opt.genFd)
    --for i,v in ipairs(dsFds) do
    --    print('directory is', v)
    --    os.execute('mkdir -p ' .. paths.concat(opt.srcGenDir, paths.basename(v)))
    --end
    os.execute('mkdir -p ' .. opt.dsGenFd)  -- parent Fd
    os.execute('mkdir -p ' .. paths.concat(opt.dsGenFd, dsSrcNm))
    local dsFdSpec = paths.concat(opt.dsGenFd, opt.dirName .. '_' .. dsSrcNm)
    local imGenFd = paths.concat(dsFdSpec, 'images')
    print('save dataset to ', dsFdSpec)
    os.execute('mkdir -p ' .. imGenFd)  -- images fd
    -- image number  imNum

    if opt.nImgGenDs > #pthsVali  then
        print('generated dataset can not exceed the number of images or valid file number')
        return
    end
    local joints_gt = torch.zeros(opt.nImgGenDs, 16,3)
    for i=1, opt.nImgGenDs do
        -- image in, joints2D in
        --local bsNm = paths.basename(pthsImg[i])
        --local img = image.load(pthsImg[i], 3, 'float')
        local pthVali = pthsVali[i]
        local pth_mp4 = pthVali:sub(1,-13) .. '.mp4'
        local img = loader.loadRGB(pth_mp4, 1)      -- only load first one
        local x_st, x_ed, idxVali = loader.getValiInfo(pth_mp4)
           -- joints2D in and process nothing to do with img now
        print('load from ', pth_mp4 )
        local joints2D = loader.loadJoints2D(pth_mp4, idxVali[math.min(idxFrm, idxVali:nElement())][1]) -- idxVali 2D tensor

        local ratio = opt.inSize[2]/opt.loadSize[2]
        joints2D:select(2,1):csub(x_st[1][1]-1 )
        joints2D = joints2D * ratio -- joints from SURREAL scale to GPM scale
        local joints_MPI = joints2D:index(1,idxMPIts):clone()
        joints_MPI[10]= joints_MPI[9] + 3*(joints_MPI[9] - joints_MPI[8])
        joints_gt[i]:narrow(2,1,2):copy(joints_MPI) -- leave last 0

        --local imgSc = cenCut(img, opt.inSize[2])  -- scaled before norm
        local imgSc, ind_st, ind_end, padDrct = sqrPadding(img, opt.inSize[2])    -- 128 stdL  imgSc original format float
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
        local imgOri = imgSc:clone()    -- source, ori padded version no normed
        -- normalization
        img = normImg(imgSc:clone())    -- source corrupted here

        -- genMapSum
        local jMap_sum = jMap:sum(1):squeeze()
       jMap_sum = jMap_sum:repeatTensor(3,1,1)     -- to 3 channels

        local jMaps = nn.utils.addSingletonDimension(jMap,1)
        local imgs = nn.utils.addSingletonDimension(img,1)

        local outputs = netG:forward({ imgs, jMaps})

        local idStg = opt.nStack  -- which stage to save out
        local output = outputs[idStg][1]   -- only one batch
        -- denorm
        output = deNormImg(output)

        image.save(paths.concat(imGenFd, string.format('image_%06d.jpg', i)), output)
    end
    joints_gt = joints_gt:permute(3,2,1)
    matio.save(paths.concat(dsFdSpec, 'joints_gt.mat'), {joints_gt = joints_gt})
end -- of test()

function testRMS()
    local idxFrm = opt.idxPose     -- always use the 60 frames
    local margin = 5
    -- set to evaluate mode
    netG:evaluate()

    local dsSrcNm = string.format(opt.dsSrcNm .. '_%d', opt.nImgGenDs)  -- SURREAL_100
    local pthsVali = dir.getallfiles(paths.concat(opt.data, opt.testDir), '*_idxVali.mat')  -- test joint infor for re position

    if opt.nImgGenDs > #pthsVali  then
        print('generated dataset can not exceed the number of images or valid file number')
        return
    end
    local RMSEtotal = {context = torch.Tensor( opt.nImgGenDs),
    rcPatch = torch.Tensor( opt.nImgGenDs)
    }

    for i=1, opt.nImgGenDs do
        -- image in, joints2D in
        --local bsNm = paths.basename(pthsImg[i])
        --local img = image.load(pthsImg[i], 3, 'float')
        local pthVali = pthsVali[i]
        local pth_mp4 = pthVali:sub(1,-13) .. '.mp4'
        local img = loader.loadRGB(pth_mp4, 1)      -- only load first one

        local x_st, x_ed, idxVali = loader.getValiInfo(pth_mp4)
        -- joints2D in and process nothing to do with img now
        print('load from ', pth_mp4 )
        local idxTar= math.min(idxFrm, idxVali:nElement())
        local joints2Dori = loader.loadJoints2D(pth_mp4,1)
        local joints2D = loader.loadJoints2D(pth_mp4, idxVali[idxTar][1]) -- idxVali 2D tensor
        local labelSc = loader.loadRGB(pth_mp4, idxTar)
        --if opt.nGPU >0 then
        --    --labelSc = labelSc:cuda()
        --end
        -- get bb
        local ratio = opt.inSize[2]/opt.loadSize[2]
        joints2D:select(2,1):csub(x_st[1][1]-1 )
        joints2D = joints2D * ratio -- joints from SURREAL scale to GPM scale
        joints2Dori:select(2,1):csub(x_st[1][1]-1 )
        joints2Dori = joints2Dori * ratio -- joints from SURREAL scale to GPM scale
        local BBori = getBB(joints2Dori:float())
        local BBtar = getBB(joints2D:float())

        local imgSc, ind_st, ind_end, padDrct = sqrPadding(img, opt.inSize[2])
        local labelSc, ind_st2, ind_end2, padDrct2 = sqrPadding(labelSc, opt.inSize[2])
        local jMap
        if 'limb' == opt.paraMap then
        jMap = loader.genMapLimbs2D(joints2D,opt.inSize[2], opt.inSize[3])
        elseif 'joint' == opt.paraMap then
        jMap = loader.genMapJoints2D(joints2D, opt.inSize[2], opt.inSize[3])
        else
        error('unknown param map type!')
        end
        local imgOri = imgSc:clone()    -- source, ori padded version no normed
        -- normalization
        img = normImg(imgSc:clone())    -- source corrupted here

        -- genMapSum
        local jMap_sum = jMap:sum(1):squeeze()
        jMap_sum = jMap_sum:repeatTensor(3,1,1)     -- to 3 channels

        local jMaps = nn.utils.addSingletonDimension(jMap,1)
        local imgs = nn.utils.addSingletonDimension(img,1)

        local outputs = netG:forward({ imgs, jMaps})

        local idStg = opt.nStack  -- which stage to save out
        local output = outputs[idStg][1]   -- only one batch
        -- denorm
        output = deNormImg(output)
        output = output:float()
        -- get masked image if i = 1 save it out
        local lbMsked = labelSc:clone()     -- float
        local outMsked = output:clone()     -- long
        lbMsked[{ {}, { BBori[2], BBori[4]}, { BBori[1], BBori[3]}}]:fill(0)   -- 2 blocks zeros
        lbMsked[{ {}, { BBtar[2], BBtar[4]}, { BBtar[1], BBtar[3]}}]:fill(0)
        outMsked[{{},{BBori[2],BBori[4]},{BBori[1],BBori[3]}}]:fill(0)
        outMsked[{{},{BBtar[2],BBtar[4]},{BBtar[1],BBtar[3]}}]:fill(0)
        local lbPatch = labelSc[{ {}, { BBori[2], BBori[4]}, { BBori[1], BBori[3]}}]:clone()
        local outPatch = output[{ {}, { BBori[2], BBori[4]}, { BBori[1], BBori[3]}}]:clone()
        -- get RMS of cropped image ( if possible, deduce the masked area)
        -- get blocked bg patch area
        -- get RMS
        --print('outMsked type', outMsked:type())
        --print('with size', outMsked:size())
        --print('lbMsked type', lbMsked:type())
        --print('lbMsked size', lbMsked:size())
        --print('outMasked is', outMsked)
        --print('label Masked is', lbMsked)
        --local diff = outMsked - lbMsked
        --print('diff is ', diff:size())
        RMSEtotal.context[i] = getRMSE(outMsked, lbMsked, true)
        RMSEtotal.rcPatch[i] = getRMSE(outPatch, lbPatch, false)
        -- add to RMStotal.context  RMStotal.recon
    end
    -- calculate and print mean and variance
    print('the mean and variance of context is',RMSEtotal.context:mean() ,RMSEtotal.context:var())
    print('the mean and variance of reconstructed area is',RMSEtotal.rcPatch:mean() ,RMSEtotal.rcPatch:var())

    -- save RMSEtotal out
    torch.save(paths.concat(opt.save,'RMSEtotal.t7'), RMSEtotal)
end -- of test()
-----------------------------------------------------------------------------

