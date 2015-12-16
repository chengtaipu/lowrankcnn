function createModel(nGPU)
   require 'cudnn'
   assert(nGPU == 1 or nGPU == 2, '1-GPU or 2-GPU supported for AlexNet')

   local fb1 = nn.Sequential() -- branch 1
   fb1:add(cudnn.SpatialConvolution(3,192,5,5,1,1,2,2))       -- 224 -> 55
   fb1:add(cudnn.ReLU())
   fb1:add(cudnn.SpatialConvolution(192,160,1,1,1,1,0,0))       -- 224 -> 55
   fb1:add(cudnn.ReLU())
   fb1:add(cudnn.SpatialConvolution(160,96,1,1,1,1,0,0))       -- 224 -> 55
   fb1:add(cudnn.ReLU())
   fb1:add(cudnn.SpatialMaxPooling(2,2,2,2))                   -- 55 ->  27
   fb1:add(nn.Dropout(0.5))
   fb1:add(cudnn.SpatialConvolution(96,192,5,5,1,1,2,2))       --  27 -> 27
   fb1:add(cudnn.ReLU())
   fb1:add(cudnn.SpatialConvolution(192,192,1,1,1,1,0,0))       --  27 -> 27
   fb1:add(cudnn.ReLU())
   fb1:add(cudnn.SpatialConvolution(192,192,1,1,1,1,0,0))       --  27 -> 27
   fb1:add(cudnn.ReLU())
   fb1:add(cudnn.SpatialAveragePooling(2,2,2,2))                   --  27 ->  13
   fb1:add(nn.Dropout(0.5))
   fb1:add(cudnn.SpatialConvolution(192,192,3,3,1,1,1,1))      --  13 ->  13
   fb1:add(cudnn.ReLU())
   fb1:add(cudnn.SpatialConvolution(192,192,1,1,1,1,0,0))      --  13 ->  13
   fb1:add(cudnn.ReLU())
   fb1:add(cudnn.SpatialConvolution(192,10,1,1,1,1,0,0))      --  13 ->  13
   fb1:add(cudnn.SpatialAveragePooling(8,8,1,1))                   -- 13 -> 6



   -- 1.3. Create Classifier (fully connected layers)
   local classifier = nn.Sequential()
   classifier:add(nn.View(10))
   classifier:add(nn.LogSoftMax())

   -- 1.4. Combine 1.1 and 1.3 to produce final model
   local model = nn.Sequential():add(fb1):add(classifier)

   return preproc, model
end
