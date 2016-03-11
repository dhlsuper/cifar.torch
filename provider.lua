vhe_reader = require 'vhe_reader'

local Provider = torch.class 'Provider'

function Provider:__init(full)
    local p_list = {}
    local n_list = {}

    local p_dir = "/Users/deweylee/Work/video_BMT/vhe/enswer/reference_8fps/"
    local n_dir = "/Users/deweylee/Work/video_BMT/vhe/enswer/query_fp/"

    for dir in io.popen([[ls "/Users/deweylee/Work/video_BMT/vhe/enswer/reference_8fps"]]):lines() do table.insert(p_list, dir) end
    for dir in io.popen([[ls "/Users/deweylee/Work/video_BMT/vhe/enswer/query_fp"]]):lines() do table.insert(n_list, dir) end

    local p_size = table.getn(p_list)
    local n_size = table.getn(n_list)

    local ptr_size = 40
    local pte_size = 4
    local ntr_size = 20
    local nte_size = 2

    local trsize = p_size*ptr_size + n_size*ntr_size
    local tesize = p_size*pte_size  + n_size*nte_size

    print(trsize, tesize)

    --process training data
    self.trainData = {
        data = torch.Tensor(trsize, 128, 128),
        labels = torch.Tensor(trsize),
        size = function() return trsize end
    }
    local trainData = self.trainData
    --negative label
    for i,v in pairs(n_list) do
        local file_name = string.format("%s%s", n_dir, v)
        print(file_name)
        trainData.data[{ {(i-1)*ntr_size+1, i*ntr_size } }] = vhe_reader.read_rand(file_name,ntr_size)
        trainData.labels[{ {(i-1)*ntr_size+1, i*ntr_size } }]:fill(2)
    end
    --positive label
    local offset = n_size*ntr_size
    for i,v in pairs(p_list) do
        local file_name = string.format("%s%s", p_dir, v)
        print(file_name)
        trainData.data[{ {offset+(i-1)*ptr_size+1, offset+i*ptr_size } }] = vhe_reader.read_rand(file_name,ptr_size)
        trainData.labels[{ {offset+(i-1)*ptr_size+1, offset+i*ptr_size } }]:fill(1)
    end
    print('training data prepared')

    --process test data
    self.testData = {
        data = torch.Tensor(tesize, 128, 128),
        labels = torch.Tensor(tesize),
        size = function() return tesize end
    }
    local testData = self.testData
    --negative label
    for i,v in pairs(n_list) do
        local file_name = string.format("%s%s", n_dir, v)
        print(file_name)
        testData.data[{ {(i-1)*nte_size+1, i*nte_size } }] = vhe_reader.read_rand(file_name,nte_size)
        testData.labels[{ {(i-1)*nte_size+1, i*nte_size } }]:fill(2)
    end
    local offset = n_size*nte_size
    for i,v in pairs(p_list) do
        local file_name = string.format("%s%s", p_dir, v)
        print(file_name)
        testData.data[{ {offset+(i-1)*pte_size+1, offset+i*pte_size } }] = vhe_reader.read_rand(file_name,pte_size)
        testData.labels[{ {offset+(i-1)*pte_size+1, offset+i*pte_size } }]:fill(1)
    end
    print('test data prepared')

    --randomize data
    print('randomizing data')
    local perm = torch.randperm(trsize):type('torch.LongTensor')
    trainData.data = trainData.data:index(1, perm)
    trainData.labels = trainData.labels:index(1, perm)
    local perm2 = torch.randperm(tesize):type('torch.LongTensor')
    testData.data = testData.data:index(1, perm2)
    testData.labels = testData.labels:index(1, perm2)

    trainData.data = trainData.data:reshape(trsize, 1, 128, 128)
    testData.data = testData.data:reshape(tesize, 1, 128, 128)

end
