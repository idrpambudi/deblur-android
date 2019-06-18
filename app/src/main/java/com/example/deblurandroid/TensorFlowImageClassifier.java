/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.example.deblurandroid;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Build;
import android.os.SystemClock;
import android.os.Trace;
import android.support.annotation.RequiresApi;
import android.util.Log;

import java.io.IOException;

import org.tensorflow.Operation;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

/** A classifier specialized to label images using TensorFlow. */
public class TensorFlowImageClassifier implements Classifier{
  private static final String TAG = "ImageClassifier";

  // Only return this many results with at least this confidence.
  private static final int MAX_RESULTS = 3;
  private static final float THRESHOLD = 0.1f;
  private static final int BATCH_SIZE = 1;
  public static final int IMG_HEIGHT = 720;
  public static final int IMG_WIDTH = 1280;
  private static final int NUM_CHANNEL = 3;
  private static final boolean IS_QUANTIZED = false;
  private static final String INPUT_NAME = "Placeholder" ;
//  private static final String OUTPUT_NAME = "g_net/dec1_0_2/BiasAdd";
  private static final String OUTPUT_NAME = "g_net/dec0_1/BiasAdd";
//  private static final String MODEL_FILE_NAME = "deblur-dw-59000-frozen-quant.pb";
  private static final String MODEL_FILE_NAME = "dw-shared-4000.pb";

  // Config values.
  private String inputName;
  private String outputName;

  // Pre-allocated buffers.
  private int[] intValues;
  private float[] outputs;
  private String[] outputNames;

  private boolean logStats = false;

  private TensorFlowInferenceInterface inferenceInterface;

  /**
   * Initializes a native TensorFlow session for classifying images.
   *
   * @param assetManager The asset manager to be used to load assets.
   * @throws IOException
   */
  public TensorFlowImageClassifier(AssetManager assetManager) throws IOException{
    inputName = INPUT_NAME;
    outputName = OUTPUT_NAME;

    inferenceInterface = new TensorFlowInferenceInterface(assetManager, MODEL_FILE_NAME);

    // The shape of the output is [N, NUM_CLASSES], where N is the batch size.
    final Operation operation = inferenceInterface.graphOperation(outputName);

    // Pre-allocate buffers.
    outputNames = new String[] {outputName};
    intValues = new int[IMG_HEIGHT * IMG_WIDTH];
    outputs = new float[IMG_HEIGHT * IMG_WIDTH * NUM_CHANNEL];
  }

  @RequiresApi(api = Build.VERSION_CODES.O)
  @Override
  public Result deblur(Bitmap bitmap) {
    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("deblurImage");

    Trace.beginSection("preprocessBitmap");
    // Preprocess the image data from 0-255 int to normalized float based
    // on the provided parameters.
//    bitmap = Bitmap.createBitmap(bitmap, 0, 0, IMG_WIDTH, IMG_HEIGHT);
    bitmap = Bitmap.createScaledBitmap(bitmap, IMG_WIDTH, IMG_HEIGHT, false);
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
    float[] floatValues = convertImageIntToFloat(intValues);
    Trace.endSection();

    // Copy the input data into TensorFlow.
    long startTime = SystemClock.uptimeMillis();
    Trace.beginSection("feed");
    inferenceInterface.feed(inputName, floatValues, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNEL);
    Trace.endSection();

    // Run the inference call.
    Trace.beginSection("run");
    inferenceInterface.run(outputNames, logStats);
    Trace.endSection();

    // Copy the output Tensor back into the output array.
    Trace.beginSection("fetch");
    inferenceInterface.fetch(outputName, outputs);
    Trace.endSection();

    Trace.endSection(); // "recognizeImage"
    long endTime = SystemClock.uptimeMillis();
    long timeCost = endTime - startTime;
    Log.v(TAG, "deblur(): timeCost = " + timeCost);

    int[] colors = convertImageFloatToInt(outputs);
    Result res = new Result(colors, timeCost, IMG_WIDTH, IMG_HEIGHT, bitmap);

    return res;
  }

  private float[] convertImageIntToFloat(int[] intArray){
    float[] floatArray = new float[intArray.length*3];
    for (int i = 0; i < intArray.length; ++i) {
      final int val = intArray[i];
      floatArray[i * 3 + 0] = ((val >> 16) & 0xFF) / 255.f;
      floatArray[i * 3 + 1] = ((val >> 8) & 0xFF) / 255.f;
      floatArray[i * 3 + 2] = (val & 0xFF) / 255.f;
    }
    return floatArray;
  }

  @RequiresApi(api = Build.VERSION_CODES.O)
  private int[] convertImageFloatToInt(float[] floatArray){
    int[] intArray = new int[floatArray.length / 3];
    for (int i = 0; i < intArray.length; ++i) {
      float r = floatArray[i * 3 + 0];
      float g = floatArray[i * 3 + 1];
      float b = floatArray[i * 3 + 2];
      int pixelValue = Color.rgb(r, g, b);
      intArray[i] = pixelValue;
    }
    return intArray;
  }

  @Override
  public void enableStatLogging(boolean logStats) {
    this.logStats = logStats;
  }

  @Override
  public String getStatString() {
    return inferenceInterface.getStatString();
  }

  @Override
  public void close() {
    inferenceInterface.close();
  }

}
