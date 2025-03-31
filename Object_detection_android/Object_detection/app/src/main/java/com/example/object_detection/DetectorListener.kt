package com.example.object_detection

interface DetectorListener {
    fun onEmptyDetect()
    fun onDetect(boundingBoxes: List<Box>, inferenceTime: Long)
}