local M = { }

function M.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Training script')
    cmd:text()
    cmd:text('Options:')  -- cmd add options, cmd:parse
    ------------ General options --------------------
    cmd:option('-dataRoot',        paths.home ..  '/datasets/', 'Home of datasets') -- not a wise version in other user case
    cmd:option('-logRoot',          '/home/jun/exp/GPM',   'home for GPM saves')
    cmd:option('-datasetname',     'cmu',           'Name of the dataset (Options: cmu')
    cmd:option('-dirName',         'GPM_test1',         'Experiment name')
    cmd:option('-outImgsDir',      'outImgs',       'Sampled images from test set ')
    cmd:option('-flgSaveIm',       true,            'if save test image out')
    cmd:option('-numOutImgs',      50,              ' how many epoch images to save from test per epoch')
    cmd:option('-serUpdtRt',       100,           ' server update rate, how many epoches to update')
    cmd:option('-data',            './',          'Path to train/test splits') -- set in main
    cmd:option('-save',            './save',      'Directory in which to log experiment') -- set in main
    cmd:option('-cache',           './cache',     'Directory in which to cache data info') -- set in main
    cmd:option('-plotter',         'plot',        'Path to the training curve.') -- set in main
    cmd:option('-trainDir',        'train',       'Directory name of the train data')
    cmd:option('-testDir',         'val',         'Directory name of the test data')
    cmd:option('-manualSeed',      1,             'Manually set RNG seed')
    cmd:option('-GPU',             1,             'Default preferred GPU')
    cmd:option('-nGPU',            1,             'Number of GPUs to use by default')
    cmd:option('-backend',         'cudnn',       'Backend')
    cmd:option('-verbose',         true,          'Verbose')
    cmd:option('-show',            false,         'Visualize input/output')
    cmd:option('-continue',        false,         'Continue stopped training')
    cmd:option('-evaluate',        false,         'Final predictions')
    cmd:option('-flgGenFd',        false,         'if generate images in specific fd')
    cmd:option('-genFd',           'samples',         'folder of the real test images')
    cmd:option('-flgGenIm',        false,         'if we generate images from certain folder ')
    cmd:option('-outGenImFd',      'outGenImFd',   'output folder of specific images, the fd name is added in main actually')
    cmd:option('-ifGenDs',        false, 'if generate DS from a specific folder')
    cmd:option('-idxPose',           20,              'the pose index to read from the idxVali, MPI just from 4, SURREAL can from middle')

    cmd:option('-dsSrcFd',        '/home/jun/SDrive/datasets_prep/GPM_src', 'GPM source folder for DS generation')
    cmd:option('-dsSrcNm',         'SURREAL',       'name of raw image folder for dataset generation')
    cmd:option('-dsGenFd',          '/home/jun/datasets/GPM_ds', 'folder for generated datasets')
    cmd:option('-nImgGenDs',        100,            'how many images to generate for dataset')
    cmd:option('-genImNm',
               --'painting/Thomas Eakins- William Rush Carving His Allegorical Figure of the Schuylkill River.png',
               'sculpture/michelangelo\'s david full body  4.jpg',
               'image name for multiple pose generations, indexes of clips are given inside')
    cmd:option('-flgGenVid',     false,           'if generate videos from images in genVidFd')
    cmd:option('-genVid',         '/home/jun/datasets/genVid',     'folder to hold the images for video generation')
    cmd:option('-outGenVid',      'outGenVid',    'folder holding the generated videos')
    cmd:option('-ifAllStgs',       0,              'if save all stages information out')
    cmd:option('-saveScores',      true,          'Score saving to txt')
    cmd:option('-outFormat',        'jpg',          'output image format')
    cmd:option('-idxSeqGen',        {17,29,38,45,47,82,120,151,375},            'the sequence idx to be generated')
    cmd:option('-ifTsRMS',        false,          'If test the RMS results and show the results')
    cmd:option('-ifABO',            0,              'if save ABO combined images')
    cmd:option('-ifASO',            1,              'if save ABS combined images')
    cmd:option('-ifCrop',           0,              'if crop the output image back to original size, for testGen parts')
    ------------- Data options ------------------------
    cmd:option('-nDonkeys',        8,             'Number of donkeys to initialize (data loading threads)')
    cmd:option('-loadSize',        {3, 240, 320}, '(#channels, height, width) of images before crop')
    cmd:option('-inSize',          {3, 128, 128}, '(#channels, height, width) of the input')
    cmd:option('-outSize',         {128, 128},      'Ground truth dimensions') -- set in main
    cmd:option('-nOutChannels',    3,            'Number of output channels') -- set in main
    cmd:option('-extension',       {'mp4'},       'Video file extensions') -- set in main
    cmd:option('-dataPtns',        {'*_idxVali.mat'},     'the data patten following linux format')
    cmd:option('-scale',           .25,           'Degree of scale augmentation')
    cmd:option('-rotate',          30,            'Degree of rotation augmentation')
    cmd:option('-supervision',     'GPM',       'Options: depth, segm')
    cmd:option('-clipsize',        100,           'Number of frames in each video clip.')
    cmd:option('-jointsIx',        {8, 5, 2, 3, 6, 9, 1, 7, 13, 16, 21, 19, 17, 18, 20, 22}, 'joints ix')
    cmd:option('-stp',             0.045,         'Depth quantization step')
    cmd:option('-depthClasses',    19,            'Number of depth bins for quantizing depth map (odd number)')
    cmd:option('-segmClasses',     15,            'Number of segmentation classes (14 body parts, +1 for background).') -- input is 24?!
    cmd:option('-paraMap',         'limb',        'the body parameter mapping methods') -- input is 24?!
    ------------- Training options --------------------
    cmd:option('-nEpochs',         50,            'Number of total epochs to run')
    cmd:option('-epochSize',       5000,          'Number of batches per epoch')
    cmd:option('-epochNumber',     1,             'Epoch number from where to train, also which for testing')
    cmd:option('-batchSize',       3,             'Mini-batch size')
    ---------- Optimization options -------consecutive---------------
    cmd:option('-optType',       'Adam',        'optimization method, Adam or RMSprop')
    cmd:option('-LR_Adam',         0.0002,        'learning rate for Adam')
    cmd:option('beta1',            0.5,          'beta1 for 1st momentum')
    cmd:option('-LR',              1e-3,          'learning rate; if set, overrides default')
    cmd:option('-momentum',        0,             'momentum')
    cmd:option('-weightDecay',     0,             'weight decay')
    cmd:option('-alpha',           0.99,          'Alpha for rmsprop')
    cmd:option('-epsilon',         1e-8,          'Epsilon for rmsprop')
    ---------- Model options ----------------------------------
    cmd:option('-netType',         'hg_gpm',       'Model type') -- set in main
    cmd:option('-ifTanh',           1,              'if use the tanh regulator for image output parts')
    cmd:option('-retrainG',         'none',        'Path to model to retrain/evaluate with netG')
    cmd:option('-retrainD',         'none',        'Path to model to retrain/evaluate with netD')
    cmd:option('-retrain_num',      0,        'which number to load')
    cmd:option('-training',        'scratch',     'Options: scratch, pretrained')
    cmd:option('-optimStateG',      'none',        'Path to an optimState to reload from')
    cmd:option('-optimStateD',      'none',        'Path to an optimState to reload from')
    cmd:option('-nStack',          3,             'Number of stacks in hg network')
    cmd:option('-nFeats',          256,           'Number of features in the hourglass')
    cmd:option('-nModules',        1,             'Number of residual modules at each location in the hourglass')
    cmd:option('-upsample',        false,         '4 times smaller output or full resolution.')
    cmd:option('-criterion',        'ABS',      'criterion between images, abs or RMS')
    cmd:option('-cGAN',             false,      'If use cGAN in this model')
    cmd:option('-D_nLayers',         2,         'n layers in D model, 2 should be better around human size') -- around > then 16 pixels  divide into 6x6 area
    cmd:option('-ndf',              64,         '#  of discrim filters in first conv layer')
    cmd:option('-use_L1',           1,         'if use L1 in cGan or not')
    cmd:option('-lambda',           100,        'weight on L1 term in objective ')
    cmd:option('-use_fcn',          0,           '1 if you want fully convolutional net for hg')
    ---------- display options ----------------------------------
    cmd:option('-display_id',       10,         'display window id ')
    cmd:option('-display_plot',     {'errG', 'errD', 'errL1'},         'loss to be displayed in result')
    cmd:text()

    local opt = cmd:parse(arg or {})

    return opt
end

return M
