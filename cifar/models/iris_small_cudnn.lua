function createModel(nGPU)
   require 'cudnn'
   assert(nGPU == 1 or nGPU == 2, '1-GPU or 2-GPU supported for AlexNet')

   local fb1 = nn.Sequential() -- branch 1

   sc=cudnn.SpatialConvolution(3,10,5,1,1,1,2,0)
   print(sc)
   fb1:add(cudnn.SpatialConvolution(3,10,1,5,1,1,0,2))
   fb1:add(cudnn.SpatialConvolution(10,128,5,1,1,1,2,0))
   fb1:add(cudnn.ReLU())
   fb1:add(cudnn.SpatialConvolution(128,128,1,1,1,1,0,0))
   fb1:add(cudnn.ReLU())
   fb1:add(nn.Dropout(0.5))
   fb1:add(cudnn.SpatialConvolution(128,10,1,1,1,1,0,0))      --  13 ->  13
   fb1:add(cudnn.SpatialAveragePooling(32,32,1,1))                   -- 13 -> 6



   -- 1.3. Create Classifier (fully connected layers)
   local classifier = nn.Sequential()
   classifier:add(nn.View(10))
   classifier:add(nn.LogSoftMax())

   -- 1.4. Combine 1.1 and 1.3 to produce final model
   local model = nn.Sequential():add(fb1):add(classifier)

   return model
end
