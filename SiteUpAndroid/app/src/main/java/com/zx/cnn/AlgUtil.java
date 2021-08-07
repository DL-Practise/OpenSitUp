package com.zx.cnn;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.util.Log;

import java.io.ByteArrayOutputStream;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.TimeUnit;

class AlgUtil implements Runnable {
    private boolean mThreadFlag;
    private final ArrayBlockingQueue mQueue;
    private final Alg mAlg;
    private final int mAlgThreads;
    private AlgCallBack mAlgCB;
    private final Thread mThread;

    public AlgUtil(AssetManager assertManager, AlgCallBack algCallBack) {
        Log.i("zhengxing", "AlgUtil::AlgUtil");
        mAlgCB = algCallBack;
        mAlgThreads = 1;
        mQueue = new ArrayBlockingQueue(3);
        mAlg = new Alg();
        mAlg.Init(assertManager);
        mThreadFlag = true;
        mThread = new Thread(this);
        mThread.start();
    }
  
    public boolean addDataToQueue(byte [] bytes, int width, int height) {
        Log.i("zhengxing", "AlgUtil::addDataToQueue");
        Bitmap bmp = null;
        try {
            YuvImage image = new YuvImage(bytes, ImageFormat.NV21, width, height, null);
            if (image != null) {
                ByteArrayOutputStream stream = new ByteArrayOutputStream();
                image.compressToJpeg(new Rect(0, 0, width, height), 100, stream);
                bmp = BitmapFactory.decodeByteArray(stream.toByteArray(), 0, stream.size());
                stream.close();
            }
        } catch (Exception ex) {
        }
        Bitmap rgba = bmp.copy(Bitmap.Config.ARGB_8888, true);
        Bitmap imgSelect = Bitmap.createScaledBitmap(rgba, 312, 312, false);
        rgba.recycle();
        return mQueue.offer(imgSelect);
    }

    public void setCallBack(AlgCallBack callBack) {
        Log.i("zhengxing", "AlgUtil::setCallBack");
        this.mAlgCB = callBack;
    }

    public void finish()
    {
        Log.i("zhengxing", "AlgUtil::finish");
        mThreadFlag = false;
    }

    @Override
    public void run() {
        Log.i("zhengxing", "AlgUtil::run");
        while (mThreadFlag) {
            try {
                Bitmap bmp = (Bitmap) mQueue.poll(1000, TimeUnit.MILLISECONDS);
                if (bmp != null) {
                    float[] x = mAlg.Run(bmp);
                    this.mAlgCB.onAlgRet(x);
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}