function createModel(nGPU)
   require 'cudnn'
   assert(nGPU == 1 or nGPU == 2, '1-GPU or 2-GPU supported for AlexNet')

   local fb1 = nn.Sequential() -- branch 1
   fb1:add(cudnn.SpatialConvolution(3,10,1,5,1,1,0,2))       -- 224 -> 55
   fb1:add(cudnn.SpatialConvolution(10,192,5,1,1,1,2,0))       -- 224 -> 55
   fb1:add(nn.SpatialBatchNormalization(192))
   fb1:add(cudnn.ReLU())
   fb1:add(cudnn.SpatialConvolution(192,160,1,1,1,1,0,0))       -- 224 -> 55
   fb1:add(cudnn.ReLU())
   fb1:add(cudnn.SpatialConvolution(160,96,1,1,1,1,0,0))       -- 224 -> 55
   fb1:add(cudnn.ReLU())

   local fb2 = nn.Sequential()
   fb2:add(cudnn.SpatialMaxPooling(2,2,2,2))                   -- 55 ->  27
   fb2:add(nn.Dropout(0.5))
   fb2:add(cudnn.SpatialConvolution(96,51,1,5,1,1,0,2))       --  27 -> 27
   fb2:add(cudnn.SpatialConvolution(51,192,5,1,1,1,2,0))       --  27 -> 27
   fb2:add(nn.SpatialBatchNormalization(192))
   fb2:add(cudnn.ReLU())
   fb2:add(cudnn.SpatialConvolution(192,192,1,1,1,1,0,0))       --  27 -> 27
   fb2:add(cudnn.ReLU())
   fb2:add(cudnn.SpatialConvolution(192,192,1,1,1,1,0,0))       --  27 -> 27
   fb2:add(cudnn.ReLU())

   local fb3 = nn.Sequential()
   fb3:add(cudnn.SpatialMaxPooling(2,2,2,2))                   --  27 ->  13
   fb3:add(nn.Dropout(0.5))
   fb3:add(cudnn.SpatialConvolution(192,192,3,3,1,1,1,1))      --  13 ->  13
   fb3:add(nn.SpatialBatchNormalization(192))
   fb3:add(cudnn.ReLU())
   fb3:add(cudnn.SpatialConvolution(192,192,1,1,1,1,0,0))      --  13 ->  13
   fb3:add(cudnn.ReLU())
   fb3:add(cudnn.SpatialConvolution(192,10,1,1,1,1,0,0))
   fb3:add(cudnn.SpatialAveragePooling(8,8,1,1))                   -- 13 -> 6


   local fb4 = nn.Sequential()
   fb4:add(fb3)
   fb4:add(cudnn.SpatialConvolution(192,10,1,1,1,1,0,0))      --  13 ->  13
   fb4:add(cudnn.SpatialAveragePooling(8,8,1,1))                   -- 13 -> 6

   local fb22 = nn.Sequential()
   fb22:add(cudnn.SpatialConvolution(96,10,1,1,1,1,0,0))
   fb22:add(cudnn.SpatialAveragePooling(32,32,1,1))                   -- 13 -> 6
   local fb33 = nn.Sequential()
   fb33:add(cudnn.SpatialConvolution(192,10,1,1,1,1,0,0))
   fb33:add(cudnn.SpatialAveragePooling(16,16,1,1))                   -- 13 -> 6

   local concat1 = nn.Concat(2)
   local concat2 = nn.Concat(2)

   concat1:add(fb22)
   concat2:add(fb3)
   concat2:add(fb33)
   concat1:add(nn.Sequential():add(fb2):add(concat2))


   
   -- 1.3. Create Classifier (fully connected layers)
   local classifier = nn.Sequential()
   classifier:add(cudnn.SpatialConvolution(30,10,1,1,1,1,0,0))
   classifier:add(nn.View(10))
   classifier:add(nn.LogSoftMax())

   -- 1.4. Combine 1.1 and 1.3 to produce final model
   local model = nn.Sequential():add(fb1):add(concat1):add(classifier)

   return preproc, model
end
