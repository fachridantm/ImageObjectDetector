package com.dicoding.picodiploma.mycamera

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.net.toUri
import com.dicoding.picodiploma.mycamera.CameraActivity.Companion.CAMERAX_RESULT
import com.dicoding.picodiploma.mycamera.databinding.ActivityMainBinding
import org.tensorflow.lite.task.vision.classifier.Classifications
import org.tensorflow.lite.task.vision.detector.Detection
import java.text.NumberFormat

class MainActivity : AppCompatActivity() {

    private lateinit var mlType: String
    private lateinit var mlHelper: MLHelper
    private lateinit var binding: ActivityMainBinding

    private var currentImageUri: Uri? = null

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            showToast("Permission request granted")
        } else {
            showToast("Permission request denied")
        }
    }

    private fun allPermissionsGranted() = ContextCompat.checkSelfPermission(
        this,
        REQUIRED_PERMISSION
    ) == PackageManager.PERMISSION_GRANTED

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        if (!allPermissionsGranted()) {
            requestPermissionLauncher.launch(REQUIRED_PERMISSION)
        }

        binding.apply {
            galleryButton.setOnClickListener { startGallery() }
            cameraButton.setOnClickListener { startCamera() }
            cameraXButton.setOnClickListener { showDialogAndExecute { startCameraX() } }
            resetImageButton.setOnClickListener {
                currentImageUri = null
                binding.previewImageView.setImageResource(R.drawable.ic_place_holder)
                binding.resultTextView.text = ""
            }
            analyzeButton.setOnClickListener {
                currentImageUri?.let {
                    showDialogAndExecute(filter = listOf(MLType.IMAGE_CLASSIFICATION)) {
                        analyzeImage(it)
                    }
                } ?: run {
                    showToast(getString(R.string.empty_image_warning))
                }
            }
        }
    }

    private fun startGallery() {
        launcherGallery.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
    }

    private val launcherGallery = registerForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri: Uri? ->
        if (uri != null) {
            currentImageUri = uri
            showImage()
        } else {
            Log.d("Photo Picker", "No media selected")
        }
    }

    private fun startCamera() {
        currentImageUri = getImageUri(this)
        launcherIntentCamera.launch(currentImageUri)
    }

    private val launcherIntentCamera = registerForActivityResult(
        ActivityResultContracts.TakePicture()
    ) { isSuccess ->
        if (isSuccess) {
            showImage()
        }
    }

    private fun startCameraX() {
        val intent = Intent(this, CameraActivity::class.java).apply {
            putExtra(CameraActivity.EXTRA_ML_TYPE, mlType)
        }
        launcherIntentCameraX.launch(intent)
    }

    private val launcherIntentCameraX = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) {
        if (it.resultCode == CAMERAX_RESULT) {
            currentImageUri = it.data?.getStringExtra(CameraActivity.EXTRA_CAMERAX_IMAGE)?.toUri()
            showImage()
        }
    }

    private fun showImage() {
        currentImageUri?.let {
            Log.d("Image URI", "showImage: $it")
            binding.previewImageView.setImageURI(it)
        }
    }

    private fun analyzeImage(uri: Uri) {
        showLoading(true)
        if (mlType == MLType.UNKNOWN.name || mlType.isEmpty()) {
            showToast("Please choose ML Type")
            showLoading(false)
            return
        }
        Log.i(TAG, "mlType: $mlType")
        mlHelper.classifyStaticImage(uri)
    }

    private fun showLoading(isLoading: Boolean) {
        binding.progressIndicator.visibility = if (isLoading) View.VISIBLE else View.GONE
    }

    private fun showToast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }

    private fun showDialogAndExecute(filter: List<MLType> = MLType.values().toList(), action: () -> Unit) {
        AlertDialog.Builder(this)
            .setTitle("Choose ML Type")
            .setItems(MLType.values().filter { filter.contains(it) }.filter { it != MLType.UNKNOWN }.map { it.getPrettyName() }.toTypedArray()) { _, which ->
                mlType = MLType.values().filter { filter.contains(it) }.filter { it != MLType.UNKNOWN }[which].name
                mlHelper = MLHelper(
                    context = this,
                    mlType = MLType.getType(mlType),
                    classifierListener = object : MLHelper.ClassifierListener {
                        override fun onError(error: String) {
                            runOnUiThread {
                                Toast.makeText(this@MainActivity, error, Toast.LENGTH_SHORT).show()
                                showLoading(false)
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
                                    } else {
                                        binding.resultTextView.text = ""
                                    }
                                }
                                showLoading(false)
                            }
                        }
                    },
                    detectorListener = object : MLHelper.DetectorListener {
                        override fun onError(error: String) {
                            runOnUiThread {
                                Toast.makeText(this@MainActivity, error, Toast.LENGTH_SHORT).show()
                                showLoading(false)
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
                                        binding.resultTextView.text = getString(R.string.detection_detected_objects, it.size)
                                    } else {
                                        binding.resultTextView.text = ""
                                    }
                                }
                                showLoading(false)
                            }
                        }

                    }
                )
                action.invoke()
            }
            .show()
    }

    companion object {
        private const val TAG = "MainActivity"
        private const val REQUIRED_PERMISSION = Manifest.permission.CAMERA
    }
}
