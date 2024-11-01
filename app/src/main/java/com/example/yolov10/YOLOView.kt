package com.example.yolov10

import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import java.util.concurrent.Executors
import kotlin.math.max


@Composable
fun CameraView(modifier: Modifier, viewModel: MainViewModel) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val cameraProvider = ProcessCameraProvider.getInstance(context).get()
    val previewView = PreviewView(context)

//    val resolutionSelector = ResolutionSelector.Builder()
//        .setResolutionFilter { supportedSizes, rotationDegrees ->
//            // 지원하는 해상도 중 1920x1080 찾기
//            supportedSizes.filter { it.width == 2560 && it.height == 1440 }
//        }
//        .build()
//
//    val preview = Preview.Builder()
//        .setResolutionSelector(resolutionSelector)
//        .build()
//        .apply { setSurfaceProvider(previewView.surfaceProvider) }
//
//    val analysis = ImageAnalysis.Builder()
//        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
//        .setResolutionSelector(resolutionSelector)
//        .build()
//        .apply {
//            setAnalyzer(Executors.newSingleThreadExecutor()) { imageProxy ->
//                // 해상도를 로그로 출력
//                val width = imageProxy.width
//                val height = imageProxy.height
//                println("Camera resolution: ${width}x${height}")
//
//                viewModel.infer(imageProxy)
//            }
//        }

    val cameraSelector = CameraSelector.Builder()
        .requireLensFacing(CameraSelector.LENS_FACING_BACK)
        .build()

    val analysis = ImageAnalysis.Builder()
        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
        .setTargetResolution(android.util.Size(1080, 1920))  // 1920x1080 해상도 설정
        .build()
        .apply {
            setAnalyzer(Executors.newSingleThreadExecutor()) { imageProxy ->
                // 해상도를 로그로 출력
                val width = imageProxy.width
                val height = imageProxy.height
                println("Camera resolution: ${width}x${height}")

                viewModel.infer(imageProxy)
            }
        }

    val preview = Preview.Builder()
        .setTargetResolution(android.util.Size(1080, 1920))  // 1920x1080 해상도 설정
        .build()
        .apply { setSurfaceProvider(previewView.surfaceProvider) }

    cameraProvider.unbindAll()
    cameraProvider.bindToLifecycle(lifecycleOwner, cameraSelector, preview, analysis)

    // 미리보기가 꽉 차지 않게 `FIT_CENTER`로 설정
    previewView.scaleType = PreviewView.ScaleType.FIT_CENTER

    AndroidView(factory = { previewView }, modifier = modifier)
}

@Composable
fun CanvasView(modifier: Modifier, viewModel: MainViewModel) {
    val resultList by viewModel.resultList.collectAsState()
    val textMeasurer = rememberTextMeasurer()
    val textStyle = TextStyle(fontSize = 15.sp)

    Canvas(modifier = modifier, onDraw = {
        resultList.forEach {
            val topLeft = Offset(it.left, it.top)
            val size = Size(it.width, it.height)

            // 바운딩 박스 그리기
            drawRect(
                color = Color.Green,
                topLeft = topLeft,
                size = size,
                style = Stroke(width = 2.dp.toPx())
            )

            // 텍스트를 그리기 전에 좌표를 검증하여 음수 값이 없도록 설정
            val textPositionX = max(0f, topLeft.x + 40) // X 좌표가 음수인지 확인 후 0 이상으로 설정
            val textPositionY = max(0f, topLeft.y)      // Y 좌표가 음수인지 확인 후 0 이상으로 설정

            // 텍스트 그리기
            drawText(
                textMeasurer = textMeasurer,
                text = "${it.className}: ${String.format("%.2f", it.confidence)}%",
                topLeft = Offset(textPositionX, textPositionY),  // 검증된 좌표로 텍스트 그리기
                style = textStyle
            )
        }
    })
}
