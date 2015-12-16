   function createModel(nGPU)
   require 'cudnn'
   assert(nGPU == 1 or nGPU == 2, '1-GPU or 2-GPU supported for AlexNet')


   local concat = nn.Concat(3)
   local fb3 = nn.Sequential()
      fb3:add(cudnn.SpatialConvolution(3,10,1,3,1,1,0,1))
      fb3:add(cudnn.SpatialConvolution(10,64,3,1,1,1,1,0))
      fb3:add(cudnn.ReLU())
    --  fb3:add(nn.SpatialBatchNormalization(64))

      fb3:add(cudnn.SpatialConvolution(64,20,1,3,1,1,0,1))
      fb3:add(cudnn.SpatialConvolution(20,64,3,1,1,1,1,0))
      fb3:add(cudnn.ReLU())
  --    fb3:add(nn.SpatialBatchNormalization(64))     

      fb3:add(cudnn.SpatialConvolution(64,10,1,1,1,1,0,0))

   local fb5  = nn.Sequential() 
      fb5:add(cudnn.SpatialConvolution(3,20,1,5,1,1,0,2))       -- 224 -> 55
      fb5:add(cudnn.SpatialConvolution(20,128,5,1,1,1,2,0))  
      fb5:add(cudnn.ReLU())
      fb5:add(cudnn.SpatialConvolution(128,40,1,5,1,1,0,2))
      fb5:add(cudnn.SpatialConvolution(40,128,5,1,1,1,2,0))
      fb5:add(cudnn.ReLU())
      fb5:add(cudnn.SpatialConvolution(128,10,1,1,1,1,0,0))
      fb5:add(cudnn.SpatialAveragePooling(32,32,1,1))
   local fb9  = nn.Sequential() 
   fb9:add(cudnn.SpatialConvolution(3,20,1,9,1,1,0,4))       -- 224 -> 55
   fb9:add(cudnn.SpatialConvolution(20,128,9,1,1,1,4,0))  
   fb9:add(cudnn.ReLU())

   fb9:add(cudnn.SpatialConvolution(128,40,1,9,1,1,0,4))
   fb9:add(cudnn.SpatialConvolution(40,128,9,1,1,1,4,0))
   fb9:add(cudnn.ReLU())

   fb9:add(cudnn.SpatialConvolution(128,10,1,1,1,1,0,0))
   fb9:add(cudnn.SpatialAveragePooling(32,32,1,1))
   concat:add(fb5)
   concat:add(fb9)
   

   local fbf = nn.Sequential()
      fbf:add(cudnn.SpatialConvolution(3,10,1,5,1,1,0,2))       -- 224 -> 55
      fbf:add(cudnn.SpatialConvolution(10,48,5,1,1,1,2,0))  
      fbf:add(cudnn.ReLU())
      fbf:add(cudnn.SpatialConvolution(48,20,1,5,1,1,0,2))
      fbf:add(cudnn.SpatialConvolution(20,48,5,1,1,1,2,0))
      fbf:add(cudnn.ReLU())
      fbf:add(cudnn.SpatialConvolution(48,10,1,9,1,1,0,4))
      fbf:add(cudnn.SpatialConvolution(10,48,9,1,1,1,4,0))  
      fbf:add(cudnn.ReLU())
      fbf:add(cudnn.SpatialConvolution(48,20,1,9,1,1,0,4))
      fbf:add(cudnn.SpatialConvolution(20,48,9,1,1,1,4,0))
      fbf:add(cudnn.ReLU())

   fbf:add(cudnn.SpatialConvolution(48,10,1,1,1,1,0,0))
   fbf:add(cudnn.SpatialAveragePooling(16,16,8,8))
   --[[fb5:add(cudnn.SpatialConvolution(3,192,5,5,1,1,2,2))       -- 224 -> 55
   fb5:add(cudnn.SpatialConvolution(3,10,1,5,1,1,0,2))       -- 224 -> 55
   fb5:add(cudnn.SpatialConvolution(10,192,5,1,1,1,2,0))       -- 224 -> 55
   fb5:add(nn.SpatialBatchNormalization(192))
   fb5:add(cudnn.ReLU())
   fb5:add(cudnn.SpatialConvolution(192,160,1,1,1,1,0,0))       -- 224 -> 55
   fb5:add(cudnn.ReLU())
   fb5:add(cudnn.SpatialConvolution(160,96,1,1,1,1,0,0))       -- 224 -> 55
   fb5:add(cudnn.ReLU())
   fb5:add(cudnn.SpatialMaxPooling(2,2,2,2))                   -- 55 ->  27
   fb5:add(nn.Dropout(0.5))
   --fb5:add(cudnn.SpatialConvolution(96,192,5,5,1,1,2,2))       --  27 -> 27
   fb5:add(cudnn.SpatialConvolution(96,51,1,5,1,1,0,2))       --  27 -> 27
   fb5:add(cudnn.SpatialConvolution(51,192,5,1,1,1,2,0))       --  27 -> 27
   fb5:add(nn.SpatialBatchNormalization(192))
   fb5:add(cudnn.ReLU())
   fb5:add(cudnn.SpatialConvolution(192,192,1,1,1,1,0,0))       --  27 -> 27
   fb5:add(cudnn.ReLU())
   fb5:add(cudnn.SpatialConvolution(192,192,1,1,1,1,0,0))       --  27 -> 27
   fb5:add(cudnn.ReLU())
   fb5:add(cudnn.SpatialAveragePooling(2,2,2,2))                   --  27 ->  13
   fb5:add(nn.Dropout(0.5))
   fb5:add(cudnn.SpatialConvolution(192,192,3,3,1,1,1,1))      --  13 ->  13
   --fb5:add(cudnn.SpatialConvolution(192,64,1,3,1,1,0,1))      --  13 ->  13
   --fb5:add(cudnn.SpatialConvolution(64,192,3,1,1,1,1,0))      --  13 ->  13
   fb5:add(nn.SpatialBatchNormalization(192))
   fb5:add(cudnn.ReLU())
   fb5:add(cudnn.SpatialConvolution(192,192,1,1,1,1,0,0))      --  13 ->  13
   fb5:add(cudnn.ReLU())
   fb5:add(cudnn.SpatialConvolution(192,10,1,1,1,1,0,0))      --  13 ->  13
   fb5:add(cudnn.SpatialAveragePooling(8,8,1,1))                   -- 13 -> 6
   --]]



   -- 1.3. Create Classifier (fully connected layers)
   local classifier = nn.Sequential()
    classifier:add(nn.View(90))
    classifier:add(nn.Linear(90,10))
    classifier:add(nn.LogSoftMax())

   -- 1.4. Combine 1.1 and 1.3 to produce final model
   local model = nn.Sequential():add(fbf):add(classifier)

   return model
   end
