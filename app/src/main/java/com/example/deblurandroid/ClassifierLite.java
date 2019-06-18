package com.example.deblurandroid;

import android.annotation.TargetApi;
import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Build;
import android.os.SystemClock;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;

public class ClassifierLite {
    private static final String LOG_TAG = ClassifierLite.class.getSimpleName();

    private static final String MODEL_NAME = "quant.tflite";
//    private static final String MODEL_NAME = "mnist.tflite";

    private static final int BATCH_SIZE = 1;
    public static final int IMG_HEIGHT = 720;
    public static final int IMG_WIDTH = 1280;
    private static final int NUM_CHANNEL = 3;
    private static final boolean IS_QUANTIZED = true;

    private final Interpreter.Options options = new Interpreter.Options();
    private final Interpreter mInterpreter;
    private final ByteBuffer mImageData;
    private final int[] mImagePixels = new int[IMG_HEIGHT * IMG_WIDTH];
    private final float[][][][] resultFloat = new float[1][IMG_HEIGHT][IMG_WIDTH][NUM_CHANNEL];
    private final byte[][][][] resultByte = new byte[1][IMG_HEIGHT][IMG_WIDTH][NUM_CHANNEL];

    private static int numBytesPerChannel;

    public ClassifierLite(Activity activity) throws IOException {
        options.setUseNNAPI(true);
        mInterpreter = new Interpreter(loadModelFile(activity), options);
        System.out.println("done creating interpreter");
        if(IS_QUANTIZED){
            numBytesPerChannel = 1;
        }
        else{
            numBytesPerChannel = 4;
        }
        mImageData = ByteBuffer.allocateDirect(
                numBytesPerChannel * BATCH_SIZE * IMG_HEIGHT * IMG_WIDTH * NUM_CHANNEL);
        mImageData.order(ByteOrder.nativeOrder());
    }

    public Result deblur(Bitmap bitmap) {
        convertBitmapToByteBuffer(bitmap);
        long startTime = SystemClock.uptimeMillis();
        if(IS_QUANTIZED){
            mInterpreter.run(mImageData, resultByte);
        }
        else{
            mInterpreter.run(mImageData, resultFloat);
        }
        long endTime = SystemClock.uptimeMillis();
        long timeCost = endTime - startTime;

        if(IS_QUANTIZED){
            Log.v(LOG_TAG, "classify(): result = " + Arrays.toString(resultByte[0])
                    + ", timeCost = " + timeCost);
            return new Result(resultByte[0], timeCost);
        }
        else{
            Log.v(LOG_TAG, "classify(): result = " + Arrays.toString(resultFloat[0])
                    + ", timeCost = " + timeCost);
            return new Result(resultFloat[0], timeCost);
        }
    }

    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        System.out.println("loading Model file");
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_NAME);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        System.out.println("done loading model file");
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (mImageData == null) {
            return;
        }
        mImageData.rewind();

        System.out.println(bitmap.getWidth() + "--" + bitmap.getHeight());
        bitmap = Bitmap.createBitmap(bitmap, 0, 0, IMG_WIDTH, IMG_HEIGHT);
        System.out.println(bitmap.getWidth() + "--" + bitmap.getHeight());
        bitmap.getPixels(mImagePixels, 0, bitmap.getWidth(), 0, 0,
                bitmap.getWidth(), bitmap.getHeight());

        int pixel = 0;
        for (int i = 0; i < IMG_WIDTH; ++i) {
            for (int j = 0; j < IMG_HEIGHT; ++j) {
                int value = mImagePixels[pixel++]; //ARGB in integer

                if(IS_QUANTIZED){
                    mImageData.put((byte) ((value >> 16) & 0xFF));
                    mImageData.put((byte) ((value >> 8) & 0xFF));
                    mImageData.put((byte) (value & 0xFF));
                }
                else{
                    mImageData.putFloat(((value >> 16) & 0xFF) / 255.f); //R
                    mImageData.putFloat(((value >> 8) & 0xFF) / 255.f); //G
                    mImageData.putFloat((value & 0xFF) / 255.f); //B
                }
            }
        }
    }


    class Result {

        private final long mTimeCost;
        private final int imgHeight;
        private final int imgWidth;
        private final Bitmap deblurImg;
        private int[] colors;

        @TargetApi(Build.VERSION_CODES.O)
        public Result(float[][][] deblurResult, long timeCost) {
            imgHeight = deblurResult.length;
            imgWidth = deblurResult[0].length;
            colors = new int[imgHeight * imgWidth];

            int colorsIndex = 0;
            for(int i = 0; i < imgWidth; i++){
                for(int j = 0; j < imgHeight; j++){
                    float red = deblurResult[i][j][0];
                    float green = deblurResult[i][j][1];
                    float blue = deblurResult[i][j][2];

                    int pixelValue = Color.rgb(red,green,blue);
                    colors[colorsIndex] = pixelValue;
                    colorsIndex++;
                }
            }
            mTimeCost = timeCost;

            deblurImg = Bitmap.createBitmap(colors, imgWidth, imgHeight, Bitmap.Config.ARGB_8888);
        }

        @TargetApi(Build.VERSION_CODES.O)
        public Result(byte[][][] deblurResult, long timeCost) {
            imgHeight = deblurResult.length;
            imgWidth = deblurResult[0].length;
            colors = new int[imgHeight * imgWidth];

            int colorsIndex = 0;
            for(int i = 0; i < imgWidth; i++){
                for(int j = 0; j < imgHeight; j++){
                    byte red = deblurResult[i][j][0];
                    byte green = deblurResult[i][j][1];
                    byte blue = deblurResult[i][j][2];

                    int pixelValue = (red & 0xff) << 16 | (green & 0xff) << 8 | (blue & 0xff);
                    colors[colorsIndex] = pixelValue;
                    colorsIndex++;
                }
            }
            mTimeCost = timeCost;

            deblurImg = Bitmap.createBitmap(colors, imgWidth, imgHeight, Bitmap.Config.ARGB_8888);
        }

        public Bitmap getDeblurResult() {
            return deblurImg;
        }

        public long getTimeCost() {
            return mTimeCost;
        }

    }

}
