typedef struct { float r, g, b, a; } Color;

__global__
void packPixels(Color* RGBInput, Color* AlphaInput, Color* Output){
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;

    Output[index].r = RGBInput[index].r;
    Output[index].g = RGBInput[index].g;
    Output[index].b = RGBInput[index].b;

    Output[index].a = AlphaInput[index].r;
}

__global__
void packPixels4(){

}

extern "C"
{
    __declspec(dllexport) Color* PackPixels(Color* RGBTexture, Color* AlphaTexture, int sizeX, int sizeY);

    //float4 RGBInput, AlphaInput, Output;
    bool isSquare;
    bool XBiggerThanY;
    int tileCount, tileSize;
    size_t bufferSize;
    Color* OutBuffer;

    Color* PackPixels(Color* RGBTexture, Color* AlphaTexture, int sizeX, int sizeY){

        // Some checks that will be useful to calculate launch parameters
        if (sizeX == sizeY){
            isSquare = true;
            XBiggerThanY = false;
            tileCount = 1;
            tileSize = sizeX;
        } else if (sizeX > sizeY){
            isSquare = false;
            XBiggerThanY = true;
            tileCount = sizeX / sizeY;
            tileSize = sizeY;
        } else {
            isSquare = false;
            XBiggerThanY = false;
            tileCount = sizeY / sizeX;
            tileSize = sizeX;
        }

        bufferSize = sizeX * sizeY * sizeof(Color);
        Color *RGBInput, *AlphaInput, *Output;

        // Allocate the textures on GPU memory
        // and the output on host
        cudaMalloc(&RGBInput, bufferSize);
        cudaMalloc(&AlphaInput, bufferSize);
        cudaMalloc(&Output, bufferSize);
        cudaMallocHost(&OutBuffer, bufferSize);

        // Copy color values to GPU memory
        cudaMemcpy(RGBInput, RGBTexture, bufferSize, cudaMemcpyHostToDevice);
        cudaMemcpy(AlphaInput, AlphaTexture, bufferSize, cudaMemcpyHostToDevice);

        dim3 grid, block;

        block = {32, 32};
        unsigned int blocksInTileSize = tileSize / 32;
        grid = {blocksInTileSize, blocksInTileSize};
        cudaStream_t streams[tileCount];

        for (int i = 0; i < tileCount; i++){
            int offset = (i) * tileSize;
            size_t count = bufferSize/tileCount;
            cudaMemcpyAsync(&RGBInput[offset], &RGBTexture[offset], count, cudaMemcpyHostToDevice, streams[i]);
            cudaMemcpyAsync(&AlphaInput[offset], &AlphaTexture[offset], count, cudaMemcpyHostToDevice, streams[i]);

            packPixels<<<grid, block, 0, streams[i]>>>(RGBInput, AlphaInput, Output);
            cudaMemcpyAsync(&OutBuffer[offset], &Output[offset], count, cudaMemcpyDeviceToHost, streams[i]);
        }

        cudaDeviceSynchronize();

        cudaFree(RGBInput);
        cudaFree(AlphaInput);
        cudaFree(Output);

        return OutBuffer;
    }

    void ClearData(){

    }
}