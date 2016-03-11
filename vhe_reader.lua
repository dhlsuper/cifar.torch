require 'struct'

local vhe_reader = {}

local function BytestoBits(nums)
    -- returns a table of bits, least significant first.
    local t={} -- will contain the bits
    for k,v in pairs(nums) do
        local num = v
        for b = 8,1,-1 do
            t[8*(k-1)+b] =math.fmod(num,2)
            num=(num-t[8*(k-1)+b])/2
        end
    end
    return t
end

function vhe_reader.read(filename)
    local inp  = io.open(filename, "rb")
    --remove header info
    local header = inp:read(12)
    header_fmt = "iiBBbb"
    local meta_info = {struct.unpack(header_fmt, header)}
    local n = meta_info[2]
    local fps = meta_info[4]
    local hop = fps/2

    local i = 1
    local data = torch.Tensor(math.floor(n/hop)+1, 128)
    for j = 1, n, hop do
        local bytes = inp:read(36)
        if not bytes then break end
        local hex = {bytes:byte(1,16)}
        bins = BytestoBits(hex)
        inp:seek("cur", 36*(hop-1))
        data[i] = torch.Tensor(bins)
        i = i+1
    end
    print('Job finished')
    
    return data
end


function vhe_reader.read_rand(filename, size)
    local inp  = io.open(filename, "rb")
    --remove header info
    local header = inp:read(12)
    header_fmt = "iiBBbb"
    local meta_info = {struct.unpack(header_fmt, header)}
    local n = meta_info[2]
    local fps = meta_info[4]
    local hop = fps/2

    local i = 1
    local data = torch.Tensor(size, 128, 128)
    for j = 1, size do
        pos = math.floor(math.random()*(n-128*hop))
        inp:seek("set", 36*(pos))
        local pic = torch.Tensor(128,128)
        for k = 1, 128 do
            local bytes = inp:read(36)
            if not bytes then break end
            local hex = {bytes:byte(1,16)}
            bins = BytestoBits(hex)
            pic[k] = torch.Tensor(bins)
            inp:seek("cur", 36*(hop-1))
        end
        data[i] = pic
        i = i+1
    end
    --print('Job finished')
    
    return data
end

return vhe_reader
