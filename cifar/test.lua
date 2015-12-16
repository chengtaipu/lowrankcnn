--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'image'
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

local testDataIterator = function()
   testLoader:reset()
   return function() return testLoader:get_batch(false) end
end

local batchNumber
local top1_center, loss
local timer = torch.Timer()
--[[
 local fb1 = nn.Sequential() -- branch 1
   fb1:add(cudnn.SpatialConvolution(3,48,5,5,1,1,2,2))       -- 224 -> 55
   fb1:add(cudnn.ReLU())
   fb1:add(cudnn.SpatialMaxPooling(2,2,2,2))                   -- 55 ->  27
   fb1:add(cudnn.SpatialConvolution(48,128,5,5,1,1,2,2))       --  27 -> 27
   fb1:add(cudnn.ReLU())
   fb1:add(cudnn.SpatialMaxPooling(2,2,2,2))                   -- 55 ->  27
   fb1:add(cudnn.SpatialConvolution(128,128,3,3,1,1,1,1)) 

   fb1:cuda()
  
fb1:get(1):share(model:get(1):get(1),'weight','bias')
fb1:get(4):share(model:get(1):get(4),'weight','bias')
fb1:get(7):share(model:get(1):get(7),'weight','bias')
--]]
function test()

   batchNumber = 0
   cutorch.synchronize()
   timer:reset()

   -- set the dropouts to evaluate mode
   model:evaluate()

   top1_center = 0
   loss = 0
   for i=1,nTest/opt.batchSize do -- nTest is set in 1_data.lua
      local indexStart = (i-1) * opt.batchSize + 1
      local indexEnd = (indexStart + opt.batchSize - 1)
      donkeys:addjob(
         -- work to be done by donkey thread
         function()
            local inputs, labels = testLoader:get(indexStart, indexEnd)
            return sendTensor(inputs), sendTensor(labels)
         end,
         -- callback that is run in the main thread once the work is done
         testBatch
      )
   end

   donkeys:synchronize()
   cutorch.synchronize()

   top1_center = top1_center * 100 / nTest
   loss = loss / (nTest/opt.batchSize) -- because loss is calculated per batch
   testLogger:add{
      ['% top1 accuracy (test set) (center crop)'] = top1_center,
      ['avg loss (test set)'] = loss
   }
   print(string.format('Epoch: [%d][TESTING SUMMARY] Total Time(s): %.2f \t'
                          .. 'average loss (per batch): %.2f \t '
                          .. 'accuracy [Center](%%):\t top-1 %.2f\t ',
                       epoch, timer:time().real, loss, top1_center))

end -- of test()
-----------------------------------------------------------------------------
local inputsCPU = torch.FloatTensor()
local labelsCPU = torch.LongTensor()
local inputs = torch.CudaTensor()
local processed_inputs = torch.CudaTensor()
local labels = torch.CudaTensor()
--local first_layer = model:get(1):get(1)
--local second_layer = model:get(1):get(2)


function testBatch(inputsThread, labelsThread)
   batchNumber = batchNumber + opt.batchSize
collectgarbage()
   receiveTensor(inputsThread, inputsCPU)
   receiveTensor(labelsThread, labelsCPU)
   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   labels:resize(labelsCPU:size()):copy(labelsCPU)
   

    local outputs = model:forward(inputs)

   --local output_1 = first_layer:forward(inputs)
   --local output_2 = second_layer:forward(output_1)
   --for i=1,192 do
    --  mean[i]:add(output_2[{{},i,{},{}}]:norm()*(output_2[{{},i,{},{}}]:norm()))
   --end

  -- local output = first_layer:forward(inputs)
  -- print(output:size())
   local err = criterion:forward(outputs, labels)
   cutorch.synchronize()
   local pred = outputs:float()

   loss = loss + err

   local _, pred_sorted = pred:sort(2, true)
   for i=1,pred:size(1) do
      local g = labelsCPU[i]
      if pred_sorted[i][1] == g then top1_center = top1_center + 1 
      end
   end
   if batchNumber % 1024 == 0 then
      --print(('Epoch: Testing [%d][%d/%d]'):format(epoch, batchNumber, nTest))
   end
end
