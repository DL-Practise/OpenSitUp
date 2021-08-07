//https://www.cnblogs.com/android100/p/android-surfaceView.html
//https://blog.csdn.net/bogongjie/article/details/84614277
package com.zx.cnn;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.graphics.PorterDuff;
import android.graphics.Rect;
import android.util.AttributeSet;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

public class CustomView extends SurfaceView implements SurfaceHolder.Callback
{
    private Context mContext;
    private final SurfaceHolder mHolder;
    private Paint mPaint;
    private int mRealWidth = 0;
    private int mRealHeight = 0;
    private float mThresHold = (float)0.1;

    private boolean mStartCount = false;
    private int mCount = 0;

    private float mHeadX = (float)(-1.0);
    private float mHeadY = (float)(-1.0);
    private float mHeadS = (float)(-1.0);
    private float mKneeX = (float)(-1.0);
    private float mKneeY = (float)(-1.0);
    private float mKneeS = (float)(-1.0);
    private float mLoinX = (float)(-1.0);
    private float mLoinY = (float)(-1.0);
    private float mLoinS = (float)(-1.0);

    private String mDebugInfo = "debug";

    // 当前状态 0=检测开始动作 1=开始计数
    //private int mState = 0;

    public CustomView(Context context) {
        super(context);
        Log.i("zhengxing", "CustomView::CustomView1");
        mContext = context;
        mHolder = this.getHolder();
        mHolder.setFormat(PixelFormat.TRANSPARENT);
        mHolder.addCallback(this);
    }

    public CustomView(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
        mContext = context;
        Log.i("zhengxing", "CustomView::CustomView2");
        mHolder = this.getHolder();
        mHolder.setFormat(PixelFormat.TRANSPARENT);
        mHolder.addCallback(this);
    }

    public void SetCount(int nCount){
        Log.i("zhengxing", "CustomView::SetCount");
        mCount = nCount;
    }

    public void drawText(String text, float x_norm, float y_norm) {
        Log.i("zhengxing", "CustomView::drawText");
        Paint paint = new Paint();
        paint.setAntiAlias(true);
        paint.setColor(Color.RED);
        paint.setTextSize(50);
        Canvas canvas = mHolder.lockCanvas();
        canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);
        canvas.drawText(text, (int)(x_norm*mRealWidth), (int)(y_norm*mRealHeight), paint);
        mHolder.unlockCanvasAndPost(canvas);
    }

    public void drawImg(Bitmap img, float x1_norm, float y1_norm, float x2_norm, float y2_norm) {
        Log.i("zhengxing", "CustomView::drawImg");
        int img_width = img.getWidth();
        int img_height = img.getHeight();
        Rect rectFrom = new Rect(0,0, img_width, img_height);
        Rect rectTo = new Rect((int)(mRealWidth*x1_norm) ,(int)(mRealHeight*y1_norm),
                                   (int)(mRealWidth*x2_norm), (int)(mRealHeight*y2_norm));
        Canvas canvas = mHolder.lockCanvas();
        canvas.drawBitmap(img, rectFrom, rectTo,null);
        mHolder.unlockCanvasAndPost(canvas);
    }

    public boolean availableDet(){
        if (mHeadS >= mThresHold && mKneeS >= mThresHold && mLoinS >= mThresHold) {
            if(mLoinX < mKneeX && mLoinY > mKneeY)
            {
                return true;
            }
        }
        return false;
    }

    public void drawResult(){
        // lock and clear the frame first
        Canvas canvas = mHolder.lockCanvas();
        canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);

        //draw the remind info
        //Bitmap img = BitmapFactory.decodeResource(mContext.getResources(), R.drawable.remind);
        //canvas.drawBitmap(img, new Rect(0,0, img.getWidth(), img.getHeight()), new Rect(0,0, mRealWidth, mRealHeight), null);

        //draw the count
        Paint paint = new Paint();
        paint.setAntiAlias(true);
        paint.setColor(Color.RED);
        paint.setTextSize(200);
        canvas.drawText("仰卧起坐计数: " + mCount, (int)(0.1*mRealWidth), (int)(0.2*mRealHeight), paint);
        //canvas.drawText(mDebugInfo, (int)(0.1*mRealWidth), (int)(0.3*mRealHeight), paint);

        if (availableDet())
        {
            paint.setStrokeWidth((float) 50.0);
            paint.setColor(Color.RED);
            canvas.drawPoint (mHeadX, mHeadY, paint);
            paint.setColor(Color.YELLOW);
            canvas.drawPoint (mKneeX, mKneeY, paint);
            paint.setColor(Color.GREEN);
            canvas.drawPoint (mLoinX, mLoinY, paint);

            paint.setColor(Color.BLUE);
            paint.setStrokeWidth((float) 20.0);
            canvas.drawLine(mHeadX, mHeadY, mLoinX, mLoinY, paint);
            canvas.drawLine(mLoinX, mLoinY, mKneeX, mKneeY, paint);
        }

        // unlock the canvas
        mHolder.unlockCanvasAndPost(canvas);
    }

    public void drawAlgRet(float[] ret) {
        int nAlgType = (int)(ret[0]);
        int nkpCount = (int) (ret[1]);
        assert(nAlgType == 2);
        assert(nkpCount == 2);
        mHeadS = ret[2 + 0*3 + 0];
        mHeadX = ((float)1.0 - ret[2 + 0*3 + 1])  * mRealWidth;
        mHeadY = ret[2 + 0*3 + 2] * mRealHeight;
        mKneeS = ret[2 + 1*3 + 0];
        mKneeX = ((float)1.0 - ret[2 + 1*3 + 1])  * mRealWidth;
        mKneeY = ret[2 + 1*3 + 2] * mRealHeight;
        mLoinS = ret[2 + 2*3 + 0];
        mLoinX = ((float)1.0 - ret[2 + 2*3 + 1])  * mRealWidth;
        mLoinY = ret[2 + 2*3 + 2] * mRealHeight;

        // alg result process
        if (availableDet())
        {
            if (mHeadY > mKneeY && mLoinY > mKneeY && mHeadX < mLoinX && mLoinX < mKneeX)
            {
                mDebugInfo = "start flag true";
                mStartCount = true;
            }
            if(mStartCount && mHeadY < mKneeY && mHeadX > mLoinX )
            {
                mCount ++;
                mStartCount = false;
                mDebugInfo = "start flag false";
            }
        }

        drawResult();
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) {
        Log.i("zhengxing", "CustomView::surfaceCreated");
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        Log.i("zhengxing", "CustomView::surfaceCreated");
        mRealWidth = width;
        mRealHeight = height;
        Canvas canvas = mHolder.lockCanvas();
        canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);
        Bitmap img = BitmapFactory.decodeResource(mContext.getResources(), R.drawable.background);
        canvas.drawBitmap(img, new Rect(0,0, img.getWidth(), img.getHeight()), new Rect(0,0, mRealWidth, mRealHeight), null);
        mHolder.unlockCanvasAndPost(canvas);
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
        Log.i("zhengxing", "CustomView::surfaceDestroyed");
    }
}  