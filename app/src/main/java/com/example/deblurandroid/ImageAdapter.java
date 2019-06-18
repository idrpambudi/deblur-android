package com.example.deblurandroid;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.annotation.NonNull;
import android.support.v4.view.PagerAdapter;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;


public class ImageAdapter extends PagerAdapter {
    private Context mContext;
    public Bitmap[] bmArray;

    ImageAdapter(Context context) {
        mContext = context;
        Bitmap bmBlur = BitmapFactory.decodeResource(context.getResources(), R.drawable.test_1);
        Bitmap bmSharp = BitmapFactory.decodeResource(context.getResources(), R.drawable.test_1_sharp);
        bmArray = new Bitmap[]{bmBlur, bmSharp};

    }

    @Override
    public int getCount() {
        return bmArray.length;
    }

    @Override
    public boolean isViewFromObject(View view, Object object) {
        return view == object;
    }

    @Override
    public Object instantiateItem(ViewGroup container, int position) {
        ImageView imageView = new ImageView(mContext);
        imageView.setScaleType(ImageView.ScaleType.CENTER_CROP);
        imageView.setImageBitmap(bmArray[position]);
        container.addView(imageView, 0);
        return imageView;
    }

    @Override
    public void destroyItem(ViewGroup container, int position, Object object) {
        container.removeView((ImageView) object);
    }

    @Override
    public int getItemPosition(@NonNull Object object) {
        return POSITION_NONE;
    }
}