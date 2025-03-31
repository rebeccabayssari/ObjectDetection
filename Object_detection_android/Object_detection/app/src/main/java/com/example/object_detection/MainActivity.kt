package com.example.object_detection

import android.Manifest

import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import android.view.MenuInflater
import android.view.MenuItem
import android.view.View
import android.widget.ImageButton
import android.widget.PopupMenu
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat


import com.example.object_detection.databinding.ActivityMainBinding

import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), DetectorListener {


    private lateinit var binding: ActivityMainBinding
    private val isFrontCamera = false

    private var preview: Preview? = null // ? utilisé pour faire un type nullable (et initié à null)
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private lateinit var detector: Detector
    private var modelName: String = "flowers_model.tflite"
    private var labelName: String = "flowers_labels.txt"
    private lateinit var cameraExecutor: ExecutorService


    // A companion is used to define static elements inside a class, owned by the class itself without need to create an instance
    companion object {
        private const val TAG = "Camera" // used for logs
        private const val CODE_PERMISSIONS = 10 //used to identify the permission requests
        private val PERMISSIONS = mutableListOf ( //List of all necessary permissions
            Manifest.permission.CAMERA
        ).toTypedArray()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)  //creates an instance of ActivityMainBinding by inflating the activity_main.xml layout file
        setContentView(binding.root) // sets the root view of this inflated layout as the content view of the activity

        val menuButton: ImageButton = findViewById(R.id.menu_button) // makes menu button responsive
        menuButton.setOnClickListener { view ->
            showPopupMenu(view) // makes menu visible
        }

        // Checks if all permissions defined in the companion are obtained and asks them if not
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, PERMISSIONS, CODE_PERMISSIONS)
        }
        detector = Detector(baseContext, modelName, labelName,this) // creates instance of the Detector with the chosen model, see **Detetcor.ktl**
        detector.setup() // used to initiate model

        cameraExecutor = Executors.newSingleThreadExecutor() // creates a new thread to manage camera recording, executes elements in the order they are given
    }


    //-------------------------------------------
    // Menu Management
    //-------------------------------------------
    private fun showPopupMenu(view: View) {
        val popup = PopupMenu(this, view) // creates an instance of the menu linked to the button (view)
        val inflater: MenuInflater = popup.menuInflater // instance of menuInflater used to inflate the resource file into popmenu
        inflater.inflate(R.menu.popup_menu, popup.menu) // inflation of the menu
        popup.setOnMenuItemClickListener { menuItem ->
            onMenuItemClick(menuItem) // makes the buttons of the menu responsive
        }
        popup.show() // show the menu
    }

    //Changes the tflite and labels used to detect elements
    private fun onMenuItemClick(item: MenuItem): Boolean {
        return when (item.itemId) {
            R.id.action_one -> {
                if (modelName != "flowers_model.tflite") {
                    modelName = "flowers_model.tflite"
                    labelName = "flowers_labels.txt"
                    detector = Detector(baseContext, modelName, labelName, this)
                    detector.setup()
                }
                true
            }
            R.id.action_two -> {
                if (modelName != "animals_model.tflite") {
                    modelName = "animals_model.tflite"
                    labelName = "animals_labels.txt"
                    detector = Detector(baseContext, modelName, labelName, this)
                    detector.setup()
                }
                true
            }
            else -> false
        }
    }

    //-------------------------------------------
    // Camera Management
    //-------------------------------------------
    private fun startCamera() {
            val cameraProviderProcess = ProcessCameraProvider.getInstance(this) // Returns a ListenableFuture<ProcessCameraProvider>
        cameraProviderProcess.addListener({ // a listener is used on the process so that once it is completely started, we can get an instance
            // The Process is ready
            cameraProvider  = cameraProviderProcess.get()
            bindCamera()
        }, ContextCompat.getMainExecutor(this)) // the listener is executed on the main thread so that it is thread safe
    }

    private fun bindCamera() {
        // Check if the camera provider has been initialized
        val cameraProvider = cameraProvider ?: throw IllegalStateException("Camera initialization failed.")

        // Determine the rotation of the display
        val rotation = binding.viewFinder.display.rotation

        // Define the camera selector to choose the back camera
        val cameraSelector = CameraSelector
            .Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .build()

        // Create a preview use case
        preview =  Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(rotation)
            .build()

        // Create an image analysis use case
        imageAnalyzer = ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setTargetRotation(binding.viewFinder.display.rotation)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()

        // Set an analyzer for the image analysis use case
        imageAnalyzer?.setAnalyzer(cameraExecutor) { imageProxy ->
            val buffer =
                Bitmap.createBitmap(
                    imageProxy.width,
                    imageProxy.height,
                    Bitmap.Config.ARGB_8888
                )

            //used to access data from a single picture
            imageProxy.use { buffer.copyPixelsFromBuffer(imageProxy.planes[0].buffer) } // copies data from the image planes to a bitmap
            imageProxy.close() // release the proxy

            // Rotate the image if needed
            val rotate = Matrix().apply {
                postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())

                // Flip the image horizontally if it's from the front camera
                if (isFrontCamera) {
                    postScale(
                        -1f,
                        1f,
                        imageProxy.width.toFloat(),
                        imageProxy.height.toFloat()
                    )
                }
            }

            // creates a new bitmap from the one generated with new options (width, height, rotation, etc.)
            val rotatedBitmap = Bitmap.createBitmap(
                buffer, 0, 0, buffer.width, buffer.height,
                rotate, true
            )

            // Perform detection on the rotated bitmap using the detector
            detector.detect(rotatedBitmap)
        }


        // Unbind all use cases from the camera provider to reset them
        cameraProvider.unbindAll() //reset de tous les binds

        try {
            // Bind the preview and image analysis use cases to the camera lifecycle
            camera = cameraProvider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalyzer
            )

            // Set the surface provider for the preview use case
            preview?.setSurfaceProvider(binding.viewFinder.surfaceProvider)

        } catch(exc: Exception) {
            // Log any exceptions that occur during use case binding
            Log.e(TAG, "Use case binding failed", exc)
        }
    }

    //-------------------------------------------
    // Activity functions (destroy, resume)
    //-------------------------------------------

    override fun onDestroy() { //called when the app is closed
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    override fun onResume() { // called when the user goes back on the app view
        super.onResume()
        if (allPermissionsGranted()){
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, PERMISSIONS, CODE_PERMISSIONS)
        }
    }

    private fun allPermissionsGranted() = PERMISSIONS.all {//function to check that all permissions defined in the companion are obtained
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    //-------------------------------------------
    // Interface Function
    //-------------------------------------------

    //We need to overload the function defined in the interface DetectorListener, used to update detection, see **Detector** and **DetectorListener**
    override fun onEmptyDetect() {
        //removes boxes shown on the screen since nothing is detected
        binding.overlay.invalidate()
    }

    override fun onDetect(boundingBoxes: List<Box>, inferenceTime: Long) {
        //adds the detected boxes to the overlay
        //UI thread is specified since detection and camera management are done on other threads
        runOnUiThread {
            binding.inferenceTime.text = "${inferenceTime}ms"
            binding.overlay.apply {
                setResults(boundingBoxes)
                invalidate()
            }
        }
    }






}