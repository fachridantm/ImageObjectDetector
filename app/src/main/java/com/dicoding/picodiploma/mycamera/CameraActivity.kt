package com.dicoding.picodiploma.mycamera

import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.View
import android.view.WindowInsets
import android.view.WindowManager
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.dicoding.picodiploma.mycamera.databinding.ActivityCameraBinding
import org.tensorflow.lite.task.vision.classifier.Classifications
import org.tensorflow.lite.task.vision.detector.Detection
import java.text.NumberFormat
import java.util.concurrent.Executors

class CameraActivity : AppCompatActivity() {
    private lateinit var binding: ActivityCameraBinding
    private lateinit var mlType: String
    private var cameraSelector: CameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        mlType = intent?.getStringExtra(EXTRA_ML_TYPE).orEmpty()

        binding = ActivityCameraBinding.inflate(layoutInflater)
        setContentView(binding.root)

    }

    public override fun onResume() {
        super.onResume()
        hideSystemUI()
        startCamera()
    }

    private fun startCamera() {
        Log.i(TAG, "mlType: $mlType")
        if (mlType == MLType.IMAGE_CLASSIFICATION.name) {
            binding.apply {
                resultTextView.visibility = View.VISIBLE
                tvInferenceTime.visibility = View.VISIBLE
            }
        } else {
            binding.apply {
                resultTextView.visibility = View.GONE
                tvInferenceTime.visibility = View.GONE
            }
        }


        val mlHelper = MLHelper(
            context = this,
            mlType = MLType.getType(mlType),
            classifierListener = object : MLHelper.ClassifierListener {
                override fun onError(error: String) {
                    runOnUiThread {
                        Toast.makeText(this@CameraActivity, error, Toast.LENGTH_SHORT).show()
                    }
                }

                override fun onResults(results: List<Classifications>?, inferenceTime: Long) {
                    runOnUiThread {
                        results?.let { it ->
                            if (it.isNotEmpty() && it[0].categories.isNotEmpty()) {
                                println(it)
                                val sortedCategories =
                                    it[0].categories.sortedByDescending { it?.score }
                                val displayResult = sortedCategories.joinToString("\n") {
                                    val score = NumberFormat.getPercentInstance().format(it.score)
                                    getString(R.string.classifiction_score_format, it.label, score)
                                }
                                binding.resultTextView.text = displayResult
                                binding.tvInferenceTime.text = getString(R.string.detection_inference_time_format, inferenceTime)
                            } else {
                                binding.resultTextView.text = ""
                                binding.tvInferenceTime.text = ""
                            }
                        }
                    }
                }
            },
            detectorListener = object : MLHelper.DetectorListener {
                override fun onError(error: String) {
                    runOnUiThread {
                        Toast.makeText(this@CameraActivity, error, Toast.LENGTH_SHORT).show()
                    }
                }

                override fun onResults(
                    results: List<Detection>?,
                    inferenceTime: Long,
                    imageHeight: Int,
                    imageWidth: Int,
                ) {
                    runOnUiThread {
                        results?.let {
                            if (it.isNotEmpty() && it[0].categories.isNotEmpty()) {
                                Log.i(TAG, "Detected ${it.size} objects")
                                binding.overlay.setResults(
                                    results, imageHeight, imageWidth
                                )
                            } else {
                                binding.overlay.clear()
                            }
                        }
                    }

                    // force a redraw
                    binding.overlay.invalidate()
                }

            }
        )

        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val resolutionSelector = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
                .build()
            val imageAnalysis = ImageAnalysis.Builder()
                .setResolutionSelector(resolutionSelector)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            val display = binding.viewFinder.display
            if (display != null) {
                imageAnalysis.targetRotation = display.rotation
            } else {
                Log.e(TAG, "ViewFinder display is null")
            }

            imageAnalysis.setAnalyzer(Executors.newSingleThreadExecutor()) { image: ImageProxy ->
                mlHelper.classifyLiveImage(image)
            }

            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.viewFinder.surfaceProvider)
            }
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalysis
            )
        }, ContextCompat.getMainExecutor(this))
    }

    private fun hideSystemUI() {
        @Suppress("DEPRECATION")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            window.insetsController?.hide(WindowInsets.Type.statusBars())
        } else {
            window.setFlags(
                WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN
            )
        }
        supportActionBar?.hide()
    }

    companion object {
        private const val TAG = "CameraActivity"
        const val EXTRA_CAMERAX_IMAGE = "CameraX Image"
        const val EXTRA_ML_TYPE = "mlType"
        const val CAMERAX_RESULT = 200
    }
}