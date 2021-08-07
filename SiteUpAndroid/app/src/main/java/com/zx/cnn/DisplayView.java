package com.zx.cnn;
import android.util.AttributeSet;
import android.util.Log;
import android.view.SurfaceView;
import android.view.SurfaceHolder;
import android.content.Context;


public class DisplayView extends SurfaceView implements SurfaceHolder.Callback
{ 
    private final SurfaceHolder mHolder;

    public DisplayView(Context context) {
        super(context);
        Log.i("zhengxing", "DisplayView::DisplayView1");
        mHolder = this.getHolder();
        mHolder.setFormat(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);
        mHolder.addCallback(this);
    }
 
    public DisplayView(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
        Log.i("zhengxing", "DisplayView::DisplayView2");
        mHolder = this.getHolder();
        mHolder.setFormat(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);
        mHolder.addCallback(this);
    }
    
    @Override
    public void surfaceCreated(SurfaceHolder holder){
        Log.i("zhengxing", "DisplayView::surfaceCreated");
    }
 
    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        Log.i("zhengxing", "DisplayView::surfaceChanged");
    }
 
    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
        Log.i("zhengxing", "DisplayView::surfaceDestroyed");
    }
}