function createModel(nGPU)
   require 'cudnn'
   assert(nGPU == 1 or nGPU == 2, '1-GPU or 2-GPU supported for AlexNet')

   local fb1 = nn.Sequential() -- branch 1
   --fb1:add(cudnn.SpatialConvolution(3,96,5,5,1,1,2,2))       -- 224 -> 55
   fb1:add(cudnn.SpatialConvolution(3,10,1,5,1,1,0,2))       -- 224 -> 55
   fb1:add(cudnn.ReLU())
   fb1:add(cudnn.SpatialConvolution(10,96,5,1,1,1,2,0))       -- 224 -> 55
   fb1:add(cudnn.ReLU())
   fb1:add(cudnn.SpatialMaxPooling(2,2,2,2))                   -- 55 ->  27
   --fb1:add(cudnn.SpatialConvolution(96,128,5,5,1,1,2,2))       --  27 -> 27
   fb1:add(cudnn.SpatialConvolution(96,32,1,5,1,1,0,2))       --  27 -> 27
   fb1:add(cudnn.ReLU())
   fb1:add(cudnn.SpatialConvolution(32,128,5,1,1,1,2,0))       --  27 -> 27
   fb1:add(cudnn.ReLU())
   fb1:add(cudnn.SpatialMaxPooling(2,2,2,2))                   --  27 ->  13
   --fb1:add(cudnn.SpatialConvolution(128,256,5,5,1,1,2,2))      --  13 ->  13
   fb1:add(cudnn.SpatialConvolution(128,64,1,5,1,1,0,2))      --  13 ->  13
   fb1:add(cudnn.ReLU())
   fb1:add(cudnn.SpatialConvolution(64,256,5,1,1,1,2,0))      --  13 ->  13
   fb1:add(cudnn.ReLU())
   fb1:add(cudnn.SpatialMaxPooling(2,2,2,2))                   -- 13 -> 6
  --[[ 
   fb1:add(cudnn.SpatialConvolution(192,192,3,3,1,1,1,1))      --  13 ->  13
   fb1:add(cudnn.ReLU(true))
   fb1:add(cudnn.SpatialConvolution(192,64,3,3,1,1,1,1))      --  13 ->  13
   fb1:add(cudnn.ReLU(true))
   fb1:add(cudnn.SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6

   --]]



   -- 1.3. Create Classifier (fully connected layers)
   local classifier = nn.Sequential()
   classifier:add(nn.View(256*4*4))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(256*4*4, 2048))
   classifier:add(nn.Threshold(0, 1e-6))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(2048, 2048))
   classifier:add(nn.Threshold(0, 1e-6))
   classifier:add(nn.Linear(2048, nClasses))
   classifier:add(nn.LogSoftMax())

   -- 1.4. Combine 1.1 and 1.3 to produce final model
   local model = nn.Sequential():add(fb1):add(classifier)

   return preproc, model
end
