package com.example.deblurandroid;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.provider.MediaStore;
import android.support.v4.view.ViewPager;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;
import android.widget.Toast;

import java.io.IOException;

import butterknife.BindView;
import butterknife.OnClick;
import butterknife.ButterKnife;

public class MainActivity extends AppCompatActivity {
    private static final String LOG_TAG = MainActivity.class.getSimpleName();
    private static final int PICK_IMAGE = 100;

    private Classifier classifier;
    private ClassifierLite mClassifier;
    private Bitmap galleryBitmap;
    private ImageAdapter adapter;

    @BindView(R.id.view_pager)
    ViewPager viewPager;
    @BindView(R.id.tv_timecost)
    TextView mTvTimeCost;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        ButterKnife.bind(this);
        init();
        Log.e(LOG_TAG, "sucecssful initialization");
    }

    private void init() {
        try {
//            mClassifier = new ClassifierLite(this);
            classifier = new TensorFlowImageClassifier(this.getAssets());
        } catch (IOException e) {
            Toast.makeText(this, R.string.failed_to_create_classifier, Toast.LENGTH_LONG).show();
            Log.e(LOG_TAG, "init(): Failed to create Classifier", e);
        }
        adapter = new ImageAdapter(this);
        viewPager.setAdapter(adapter);
    }

    @OnClick(R.id.btn_deblur)
    void onDeblurClick() {
        if (classifier == null) {
            Log.e(LOG_TAG, "onDeblurClick(): Classifier is not initialized");
            return;
        }
        Bitmap image = adapter.bmArray[0];
        Classifier.Result result = classifier.deblur(image);
        adapter.bmArray[1] = result.getDeblurResult();
        mTvTimeCost.setText(String.format(getString(R.string.timecost_value), result.getTimeCost()));
        adapter.notifyDataSetChanged();
    }

    @OnClick(R.id.btn_search)
    void onSearchClick() {
        Intent galleryIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.INTERNAL_CONTENT_URI);

        startActivityForResult(galleryIntent, PICK_IMAGE);
        adapter.bmArray[0] = galleryBitmap;
        adapter.notifyDataSetChanged();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(resultCode == RESULT_OK && requestCode == PICK_IMAGE){
            try {
                Uri imageUri = data.getData();
                galleryBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageUri);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
