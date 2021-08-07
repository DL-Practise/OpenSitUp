package com.zx.cnn;
import android.hardware.Camera;
import android.util.Log;
import android.view.SurfaceView;

import java.io.*;


public class CameraUtil
{
    private Camera mCamera;
    private final int mCameraID;
    private final SurfaceView mViewDisplay;
    private final int mOrientation;
    private final Camera.PreviewCallback mPreviewCBack;
    
    public CameraUtil(SurfaceView displayView, Camera.PreviewCallback cameraCBack) {
        Log.i("zhengxing", "CameraUtil::CameraUtil");
        mCamera = null;
        mViewDisplay = displayView;
        mCameraID = Camera.CameraInfo.CAMERA_FACING_FRONT;
        mOrientation = 0;
        mPreviewCBack = cameraCBack;
    }
        
    public int getCameraState(){
        Log.i("zhengxing", "CameraUtil::CameraUtil");
        if(mCamera == null)
        {
            return -1; //camera is off
        }
        else
        {
            return mOrientation; //return camera id
        }
    }
        
    public void openCamera() {
        Log.i("zhengxing", "CameraUtil::openCamera");
        if(mCamera == null) {
            mCamera = Camera.open(mCameraID);
            Camera.Parameters parameters = mCamera.getParameters();
            //parameters.setFocusMode(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO);
            //mCamera.setParameters(parameters);
            mCamera.setDisplayOrientation(mOrientation);
            mCamera.setPreviewCallback(mPreviewCBack);
            try {
                mCamera.setPreviewDisplay(mViewDisplay.getHolder());
            } catch (IOException e) {
                e.printStackTrace();
            }
            mCamera.startPreview();
        }
    }
    
    public void closeCamera() {
        Log.i("zhengxing", "CameraUtil::closeCamera");
        if (mCamera != null) {
            mCamera.setPreviewCallback(null);
            mCamera.stopPreview();
            mCamera.release();
            mCamera = null;
        }
    }
    
    public void changeCamera() {
        Log.i("zhengxing", "CameraUtil::closeCamera");
    }
}