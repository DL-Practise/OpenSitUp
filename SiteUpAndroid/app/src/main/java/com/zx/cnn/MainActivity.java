package com.zx.cnn;
import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.hardware.Camera;
import android.net.Uri;
import android.os.Bundle;
import android.content.pm.ActivityInfo;
import android.util.Log;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;

public class MainActivity extends Activity implements Camera.PreviewCallback, AlgCallBack{

    private DisplayView  mViewDisplay;
    private CustomView   mViewCustom;
    private Button       mBtnCameraOp;
    private Button       mBtnCameraChange;
    private CameraUtil   mCameraUtil;
    private AlgUtil      mAlgUtil;
    private int          mFrameCount = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Log.i("zhengxing", "MainActivity::onCreate");
        // set the basic
        super.onCreate(savedInstanceState);
        this.getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);
        this.requestWindowFeature(Window.FEATURE_NO_TITLE);
        this.getWindow().setFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON, WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
        setContentView(R.layout.main);
        Log.i("zhengxing", "MainActivity::onCreate set basic info finished");
        
        // create visual toolkits
        mViewDisplay = (DisplayView)this.findViewById(R.id.display_view);
        mViewCustom = (CustomView)this.findViewById(R.id.custom_view);
        Log.i("zhengxing", "MainActivity::onCreate create visual toolkits finished");
        
        // init the utils
        mCameraUtil = new CameraUtil(mViewDisplay, this);
        mAlgUtil = new AlgUtil(getAssets(), this);
        Log.i("zhengxing", "MainActivity::onCreate create camera util and alg util finished");

        // get the permissions
        if (this.checkSelfPermission(Manifest.permission.CAMERA)
                != this.getPackageManager().PERMISSION_GRANTED) {
            this.requestPermissions(new String[]{Manifest.permission.CAMERA}, 1);
        }
        if (this.checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != this.getPackageManager().PERMISSION_GRANTED) {
            this.requestPermissions(new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, 1);
        }
    }


    // button camera op callback
    public void onBtnStartClick(View view){
        Log.i("zhengxing", "MainActivity::onBtnStartClick");
        if (mCameraUtil.getCameraState() < 0){
            mCameraUtil.openCamera();
            Log.i("zhengxing", "MainActivity::onBtnStartClick the camera is closed, open it");
        }
    }

    public void onBtnStopClick(View view){
        Log.i("zhengxing", "MainActivity::onBtnStopClick");
        if (mCameraUtil.getCameraState() >= 0){
            mCameraUtil.closeCamera();
            Log.i("zhengxing", "MainActivity::onBtnStopClick the camera is open, close it");
        }
    }

    public void onBtnClearClick(View view){
        Log.i("zhengxing", "MainActivity::onBtnClearClick");
        mViewCustom.SetCount(0);
    }

    public void onBtnHelpClick(View view){
        Log.i("zhengxing", "MainActivity::onBtnHelpClick");
        Bitmap img = BitmapFactory.decodeResource(this.getResources(), R.drawable.help);
        mViewCustom.drawImg(img, (float)0.0, (float)0.0, (float)1.0, (float)1.0);
    }

    public void onBtnCommitClick(View view){
        Log.i("zhengxing", "MainActivity::onBtnCommitClick");
        //Bitmap img = BitmapFactory.decodeResource(this.getResources(), R.drawable.commit);
        //mViewCustom.drawImg(img, (float)0.0, (float)0.0, (float)1.0, (float)1.0);

        Log.i("zhengxing", "MainActivity::onBtnCommitClick");
        Intent intent = new Intent();
        intent.setAction("android.intent.action.VIEW");
        intent.addCategory("android.intent.category.BROWSABLE");
        intent.addCategory("android.intent.category.DEFAULT");
        Uri content_url = Uri.parse("http://www.lgddx.cn/xmlb/siteup_count/ask_url");
        intent.setData(content_url);
        startActivity(intent);
    }

    // alg callback
    @Override
    public void onAlgRet(float[] ret) {
        float nAlgType = ret[0];
        float nClasss = ret[1];
        Log.i("zhengxing", "MainActivity::onAlgRet ret value:" +  ret[0] + ';' + ret[1]);
        mViewCustom.drawAlgRet(ret);
    }

    // preview callback
    @Override
    public void onPreviewFrame(byte[] data, Camera camera) {
        mFrameCount ++;
        Log.i("zhengxing", "MainActivity::onPreviewFrame");
        //mViewCustom.drawText("frame seq: " + mFrameCount, 300, 100);
        Camera.Size size = camera.getParameters().getPreviewSize();
        mAlgUtil.addDataToQueue(data, size.width, size.height);
    }
    
}
