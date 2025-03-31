package com.example.object_detection

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader

class   Detector(
    private val context: Context,
    private val modelPath: String,
    private val labelPath: String,
    private val detectorListener: DetectorListener
) {

    private var interpreter: Interpreter? = null
    private var labels = mutableListOf<String>()

    private var tensorWidth = 0
    private var tensorHeight = 0
    private var numChannel = 0
    private var numElements = 0

    //creates pipeline to treat images before feeding them to a tflite model
    //normalizes the pixel values of the image and standard deviation
    //also ensures that data used is in the right format (Float32 for example)
    private val imageProcessor = ImageProcessor.Builder()
        .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
        .add(CastOp(INPUT_IMAGE_TYPE))
        .build()

    //initializes the model used for detection
    fun setup() {
        val model =
            FileUtil.loadMappedFile(context, modelPath) //loads the model file stored in /assets
        val options =
            Interpreter.Options() //creates an instance of Interpreter.Options that will be given to the interpreter
        options.numThreads = 4 //sets number of threads used to analyze image
        interpreter =
            Interpreter(model, options) // creates instance of interpreter with the loaded model

        val inputShape =
            interpreter?.getInputTensor(0)?.shape() ?: return //retrieves input parameters specs
        val outputShape = interpreter?.getOutputTensor(0)?.shape() ?: return // idem for output

        tensorWidth = inputShape[1] // image width
        tensorHeight = inputShape[2] // image height
        numChannel = outputShape[1] //defines the number of classes that the model can detect
        numElements =
            outputShape[2] //defines the number of elements that can be detected at once in an image

        try {
            //used to read the label file given as an argument in the detector constructor
            val inputStream: InputStream = context.assets.open(labelPath)
            val reader = BufferedReader(InputStreamReader(inputStream))

            //reads each line and adds it to the labels (list of strings)
            var line: String? = reader.readLine()
            while (line != null && line != "") {
                labels.add(line)
                line = reader.readLine()
            }

            reader.close()
            inputStream.close()
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }

    fun detect(frame: Bitmap) {

        //returns if a parameter os wrong (null or zero)
        interpreter ?: return
        if (tensorWidth == 0) return
        if (tensorHeight == 0) return
        if (numChannel == 0) return
        if (numElements == 0) return

        var inferenceTime =
            SystemClock.uptimeMillis() // records the current time in ms, used to measure time taken for the inference process

        val resizedBitmap = Bitmap.createScaledBitmap(
            frame,
            tensorWidth,
            tensorHeight,
            false
        ) // resizes Bitmap to match parameters of model

        val tensorImage =
            TensorImage(DataType.FLOAT32) // transforms image to make it compatible with a tensorflow model with specific data
        tensorImage.load(resizedBitmap)
        val processedImage =
            imageProcessor.process(tensorImage) // image processed (see comment when processor is declared)
        val imageBuffer = processedImage.buffer //retrieves processed data image

        val output = TensorBuffer.createFixedSize(
            intArrayOf(1, numChannel, numElements),
            OUTPUT_IMAGE_TYPE
        ) // creates a specific buffer to manage image output
        interpreter?.run(imageBuffer, output.buffer) //runs inference on the input image data

        val bestBoxes = bestBox(output.floatArray) // processes the raw output data
        inferenceTime =
            SystemClock.uptimeMillis() - inferenceTime //calculates the time taken to infer

        // if no output is given, goes back in detection mode
        if (bestBoxes == null) {
            detectorListener.onEmptyDetect()
            return
        }

        //action when detected, see **DetectorListener interface**
        detectorListener.onDetect(bestBoxes, inferenceTime)
    }

    private fun bestBox(array: FloatArray): List<Box>? {

        //used to find the best results from output data of the model
        val boundingBoxes = mutableListOf<Box>()

        //For each element detected on the image, we check the class for which the confidence rate is the highest
        for (c in 0 until numElements) {
            var maxConf = -1.0f
            var maxIdx = -1
            var j =
                4 // initialized to 4 since the four first variables for a detected element are the coordinates,
            // allows to reach directly the confidence rates of the different classes
            var arrayIdx =
                c + numElements * j //used to access the max confidence of the given element
            while (j < numChannel) {
                if (array[arrayIdx] > maxConf) {
                    maxConf = array[arrayIdx]
                    maxIdx = j - 4 //gives the beginning of the elements about the current element
                }
                j++
                arrayIdx += numElements
            }

            if (maxConf > CONFIDENCE_THRESHOLD) {
                val clsName = labels[maxIdx] //we get the name of the class

                //coordiantes of the box center and size:
                val cx = array[c]
                val cy = array[c + numElements]
                val w = array[c + numElements * 2]
                val h = array[c + numElements * 3]

                //coordinates of the box edges:
                val x1 = cx - (w / 2F)
                val y1 = cy - (h / 2F)
                val x2 = cx + (w / 2F)
                val y2 = cy + (h / 2F)
                //conditions to be sure that the bix is inside the screen
                if (x1 < 0F || x1 > 1F) continue
                if (y1 < 0F || y1 > 1F) continue
                if (x2 < 0F || x2 > 1F) continue
                if (y2 < 0F || y2 > 1F) continue

                //add the box for the element with the maximum confidence rate
                boundingBoxes.add(
                    Box(
                        x1 = x1, y1 = y1, x2 = x2, y2 = y2,
                        cx = cx, cy = cy, w = w, h = h,
                        cnf = maxConf, cls = maxIdx, clsName = clsName
                    )
                )
            }
        }

        if (boundingBoxes.isEmpty()) return null

        return applyNMS(boundingBoxes)
    }

    //Allows to prevent box superposition if several elemnts are detected
    private fun applyNMS(boxes: List<Box>): MutableList<Box> {
        val sortedBoxes = boxes.sortedByDescending { it.cnf }.toMutableList()
        val selectedBoxes = mutableListOf<Box>()

        while (sortedBoxes.isNotEmpty()) {
            //we begin by sorting the boxes, the first one having the best confidance rate (the most reliable)
            val first = sortedBoxes.first()
            selectedBoxes.add(first)
            sortedBoxes.remove(first)

            //Then we check if an other boc intersects with the first one
            val iterator = sortedBoxes.iterator()
            while (iterator.hasNext()) {
                val nextBox = iterator.next()
                val iou = calculateIoU(first, nextBox)
                if (iou >= IOU_THRESHOLD) {
                    iterator.remove()
                }
            }
        }

        return selectedBoxes
    }

    // Allows to calculate if two boxes intersect by calculating the intersection over union
    private fun calculateIoU(box1: Box, box2: Box): Float {
        val x1 = maxOf(box1.x1, box2.x1)
        val y1 = maxOf(box1.y1, box2.y1)
        val x2 = minOf(box1.x2, box2.x2)
        val y2 = minOf(box1.y2, box2.y2)
        val intersectionArea = maxOf(0F, x2 - x1) * maxOf(0F, y2 - y1)
        val box1Area = box1.w * box1.h
        val box2Area = box2.w * box2.h
        return intersectionArea / (box1Area + box2Area - intersectionArea)
    }

    companion object {
        private const val INPUT_MEAN = 0f
        private const val INPUT_STANDARD_DEVIATION = 255f
        private val INPUT_IMAGE_TYPE = DataType.FLOAT32
        private val OUTPUT_IMAGE_TYPE = DataType.FLOAT32
        private const val CONFIDENCE_THRESHOLD = 0.3F
        private const val IOU_THRESHOLD = 0.5F
    }
}