package com.dicoding.picodiploma.mycamera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.SystemClock
import android.provider.MediaStore
import android.util.Log
import android.view.Surface
import androidx.camera.core.ImageProxy
import com.dicoding.picodiploma.mycamera.MLHelper.Companion.IMAGE_CLASSIFICATION_MODEL
import com.dicoding.picodiploma.mycamera.MLHelper.Companion.OBJECT_DETECTION_MODEL
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.core.vision.ImageProcessingOptions
import org.tensorflow.lite.task.vision.classifier.Classifications
import org.tensorflow.lite.task.vision.classifier.ImageClassifier
import org.tensorflow.lite.task.vision.detector.Detection
import org.tensorflow.lite.task.vision.detector.ObjectDetector

class MLHelper(
    private var threshold: Float = 0.1f,
    private var maxResults: Int = 3,
    private val mlType: MLType,
    val context: Context,
    val classifierListener: ClassifierListener?,
    val detectorListener: DetectorListener?
) {
    private var imageClassifier: ImageClassifier? = null
    private var objectDetector: ObjectDetector? = null

    init {
        when (mlType) {
            MLType.IMAGE_CLASSIFICATION -> setupImageClassifier()
            MLType.OBJECT_DETECTION -> setupObjectDetector()
            MLType.UNKNOWN -> throw IllegalArgumentException("Unknown MLType")
        }
    }

    private fun setupImageClassifier() {
        val optionsClassifierBuilder = ImageClassifier.ImageClassifierOptions.builder()
            .setScoreThreshold(threshold)
            .setMaxResults(maxResults)
        val optionsDetectorBuilder = ObjectDetector.ObjectDetectorOptions.builder()
            .setScoreThreshold(threshold)
            .setMaxResults(maxResults)
        val baseOptionsBuilder = BaseOptions.builder()
            .setNumThreads(4)
        optionsClassifierBuilder.setBaseOptions(baseOptionsBuilder.build())
        optionsDetectorBuilder.setBaseOptions(baseOptionsBuilder.build())

        try {
            imageClassifier = ImageClassifier.createFromFileAndOptions(
                context,
                mlType.modelName,
                optionsClassifierBuilder.build()
            )
        } catch (e: IllegalStateException) {
            classifierListener?.onError(context.getString(R.string.image_classifier_failed))
            Log.e(TAG, e.message.orEmpty())
        }
    }

    private fun setupObjectDetector() {
        val optionsBuilder = ObjectDetector.ObjectDetectorOptions.builder()
            .setScoreThreshold(threshold)
            .setMaxResults(maxResults)
        val baseOptionsBuilder = BaseOptions.builder()
            .setNumThreads(4)
        optionsBuilder.setBaseOptions(baseOptionsBuilder.build())

        try {
            objectDetector = ObjectDetector.createFromFileAndOptions(
                context,
                mlType.modelName,
                optionsBuilder.build()
            )
        } catch (e: IllegalStateException) {
            classifierListener?.onError(context.getString(R.string.image_classifier_failed))
            Log.e(TAG, e.message.orEmpty())
        }
    }

    fun classifyStaticImage(imageUri: Uri) {
        imageClassifier?.let { setupImageClassifier() }

        val imageProcessor = ImageProcessor.Builder().build()
        val bitmapUri = getBitmapFromUri(context, imageUri)

        bitmapUri?.copy(Bitmap.Config.ARGB_8888, true)?.let { bitmap ->
            val tensorImage = imageProcessor.process(TensorImage.fromBitmap(bitmap))

            val imageProcessingOptions = ImageProcessingOptions.builder()
                .build()

            var inferenceTime = SystemClock.uptimeMillis()
            val results = imageClassifier?.classify(tensorImage, imageProcessingOptions)
            inferenceTime = SystemClock.uptimeMillis() - inferenceTime
            classifierListener?.onResults(
                results,
                inferenceTime
            )
        }
    }

    fun classifyLiveImage(image: ImageProxy) {

        when (mlType) {
            MLType.IMAGE_CLASSIFICATION -> imageClassifier?.let { setupImageClassifier() }
            MLType.OBJECT_DETECTION -> objectDetector?.let { setupObjectDetector() }
            MLType.UNKNOWN -> throw IllegalArgumentException("Unknown MLType")
        }

        val imageProcessorBuilder = ImageProcessor.Builder()

        // Example operations, ensure they are not null
        val resizeOp = ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR)
        val normalizeOp = NormalizeOp(0.0f, 1.0f)

        imageProcessorBuilder.add(resizeOp)
        imageProcessorBuilder.add(normalizeOp)

        val imageProcessor = imageProcessorBuilder.build()

        val bitmapBuffer = Bitmap.createBitmap(
            image.width,
            image.height,
            Bitmap.Config.ARGB_8888
        )
        image.use { bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer) }
        image.close()

        val tensorImage: TensorImage = imageProcessor.process(TensorImage.fromBitmap(bitmapBuffer))

        val imageProcessingOptions = ImageProcessingOptions.builder()
            .setOrientation(getOrientationFromRotation(image.imageInfo.rotationDegrees))
            .build()

        var inferenceTime = SystemClock.uptimeMillis()
        val resultsClassifier = imageClassifier?.classify(tensorImage, imageProcessingOptions)
        val resultsDetector = objectDetector?.detect(tensorImage, imageProcessingOptions)
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime

        when(mlType) {
            MLType.IMAGE_CLASSIFICATION -> classifierListener?.onResults(
                resultsClassifier,
                inferenceTime
            )

            MLType.OBJECT_DETECTION -> {
                detectorListener?.onResults(
                    resultsDetector,
                    inferenceTime,
                    tensorImage.height,
                    tensorImage.width
                )
            }

            MLType.UNKNOWN -> throw IllegalArgumentException("Unknown MLType")
        }
    }

    private fun getOrientationFromRotation(rotation: Int): ImageProcessingOptions.Orientation {
        return when (rotation) {
            Surface.ROTATION_270 -> ImageProcessingOptions.Orientation.BOTTOM_RIGHT
            Surface.ROTATION_180 -> ImageProcessingOptions.Orientation.RIGHT_BOTTOM
            Surface.ROTATION_90 -> ImageProcessingOptions.Orientation.TOP_LEFT
            else -> ImageProcessingOptions.Orientation.RIGHT_TOP
        }
    }

    interface ClassifierListener {
        fun onError(error: String)
        fun onResults(
            results: List<Classifications>?,
            inferenceTime: Long
        )
    }

    interface DetectorListener {
        fun onError(error: String)
        fun onResults(
            results: List<Detection>?,
            inferenceTime: Long,
            imageHeight: Int,
            imageWidth: Int
        )
    }

    companion object {
        const val OBJECT_DETECTION_MODEL = "efficientdet_lite0_v1.tflite"
        const val IMAGE_CLASSIFICATION_MODEL = "mobilenet_v1.tflite"
        private const val TAG = "MLHelper"
    }
}

enum class MLType(val modelName: String) {
    OBJECT_DETECTION(OBJECT_DETECTION_MODEL),
    IMAGE_CLASSIFICATION(IMAGE_CLASSIFICATION_MODEL),
    UNKNOWN("unknown")
    ;

    fun getPrettyName() = name.replace("_", " ").lowercase().split(" ").joinToString(" ") { it.replaceFirstChar { it.uppercase() } }

    companion object {
        fun getType(name: String): MLType {
            return when (name.uppercase()) {
                OBJECT_DETECTION.name -> OBJECT_DETECTION
                IMAGE_CLASSIFICATION.name -> IMAGE_CLASSIFICATION
                else -> UNKNOWN
            }
        }
    }
}