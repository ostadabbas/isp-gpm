-- From https://github.com/anewell/pose-hg-train

local conv = cudnn.SpatialConvolution
local decov = cudnn.SpatialFullConvolution
local batchnorm = nn.SpatialBatchNormalization
local relu = cudnn.ReLU
local deconv = cudnn.SpatialFullConvolution

-- Main convolutional block
local function convBlock(numIn,numOut)  -- all steps 1 1 same size
    return nn.Sequential()
        :add(batchnorm(numIn))
        :add(relu(true))
        :add(conv(numIn,numOut/2,1,1))
        :add(batchnorm(numOut/2))
        :add(relu(true))
        :add(conv(numOut/2,numOut/2,3,3,1,1,1,1))   -- step 1,1, pad 1,1 
        :add(batchnorm(numOut/2))
        :add(relu(true))
        :add(conv(numOut/2,numOut,1,1))
end

function deconvBlock(numIn,numOut)  -- all steps 1 1 same size, see outside
    return nn.Sequential()
        :add(batchnorm(numIn))
        :add(relu(true))
        :add(deconv(numIn, numOut, 4, 4, 2, 2, 1,1))
end

-- Skip layer
local function skipLayer(numIn,numOut)
    if numIn == numOut then
        return nn.Identity()
    else
        return nn.Sequential()
            :add(conv(numIn,numOut,1,1))
    end
end

-- Residual block
function Residual(numIn,numOut)
    return nn.Sequential()
        :add(nn.ConcatTable()
            :add(convBlock(numIn,numOut))
            :add(skipLayer(numIn,numOut)))
        :add(nn.CAddTable(true))
end

