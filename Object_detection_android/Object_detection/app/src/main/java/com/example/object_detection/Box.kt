package com.example.object_detection

//characteristics used to draw the boxes on the canvas defined in Overlay to show a detected element
data class Box(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val cx: Float,
    val cy: Float,
    val w: Float,
    val h: Float,
    val cnf: Float,
    val cls: Int,
    val clsName: String
)
