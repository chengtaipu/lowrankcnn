function createModel(nGPU)
   require 'cudnn'
   assert(nGPU == 1 or nGPU == 2, '1-GPU or 2-GPU supported for AlexNet')

local baseline = nn.Sequential()
      baseline:add(cudnn.SpatialConvolution(3,10,1,5,1,1,0,2))
      baseline:add(cudnn.SpatialConvolution(10,96,5,1,1,1,2,0))      
      baseline:add(cudnn.ReLU())
      baseline:add(cudnn.SpatialMaxPooling(2,2,2,2))
      baseline:add(cudnn.SpatialConvolution(96,51,1,5,1,1,0,2))
      baseline:add(cudnn.SpatialConvolution(51,128,5,1,1,1,2,0))
      baseline:add(cudnn.ReLU())
      baseline:add(cudnn.SpatialMaxPooling(2,2,2,2))
      baseline:add(cudnn.SpatialConvolution(128,51,1,5,1,1,0,2))
      baseline:add(cudnn.SpatialConvolution(51,256,5,1,1,1,2,0))
      baseline:add(cudnn.ReLU())
      baseline:add(cudnn.SpatialMaxPooling(2,2,2,2))
      baseline:add(cudnn.SpatialConvolution(256,64,1,1,1,1,0,0))
      baseline:add(cudnn.ReLU())



   -- 1.3. Create Classifier (fully connected layers)
   local classifier = nn.Sequential()
   classifier:add(nn.View(1024))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(1024, 256))
   classifier:add(nn.Threshold(0, 1e-6))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(256, nClasses))
   classifier:add(nn.LogSoftMax())
   -- 1.4. Combine 1.1 and 1.3 to produce final model
   local model = nn.Sequential():add(baseline):add(classifier)

   return model
end
