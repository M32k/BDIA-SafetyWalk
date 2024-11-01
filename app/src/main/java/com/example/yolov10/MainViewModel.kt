package com.example.yolov10

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.res.Resources
import android.graphics.Bitmap
import android.graphics.Color
import androidx.camera.core.ImageProxy
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Collections
import kotlin.math.max
import kotlin.math.min
import android.graphics.Matrix
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll

class MainViewModel(private val resources: Resources) : ViewModel() {
    private val _resultList = MutableStateFlow(mutableListOf<Result>())
    val resultList: StateFlow<MutableList<Result>> = _resultList.asStateFlow()

    private val confidenceThreshold = 0.75f
    private val ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val ortSession: OrtSession = ortEnv.createSession(readModel())
    private val classes = readClasses()

    private val shape = (ortSession.inputInfo["images"]?.info as TensorInfo).shape
    private val width = shape[3].toInt()
    private val height = shape[2].toInt()
    private val imageStd = 255f

    private var dx = 1f
    private var dy = 1f
    private var diffY = 0f
    private var phoneWidth = 1
    private var phoneHeight = 1

    fun infer(imageProxy: ImageProxy) = viewModelScope.launch(Dispatchers.Default) {
        // 잘린 6개의 이미지를 얻어옴
        val bitmapsWithOffsets = preProcessMultipleCroppedImages(imageProxy)

        // 각 타일의 바운딩 박스를 추출한 후 전체 이미지 좌표로 변환하고, 모든 결과를 하나의 리스트에 저장
        val allResults = mutableListOf<Result>()

        // 각 이미지를 비동기로 처리하여 결과를 병렬로 계산
        val deferredResults = bitmapsWithOffsets.map { (bitmap, offsetX, offsetY) ->
            async {
                val inputTensor = preProcessBitmap(bitmap)
                val rawOutput = process(inputTensor)
                // 각 타일에서 얻은 바운딩 박스에 오프셋 적용하여 전체 이미지 좌표로 변환
                postProcess(rawOutput, offsetX, offsetY)
            }
        }

        // 모든 결과를 한 번에 취합
        val tileResults = deferredResults.awaitAll().flatten().toMutableList()

        // 타일별 바운딩 박스들을 전체적으로 취합한 후 NMS 적용
        allResults.addAll(tileResults)

        // NMS로 중복 바운딩 박스 제거
        val finalResults = applyNMS(allResults, 0.3f)  // IoU 임계값 0.3

        // 최종 결과 emit
        _resultList.emit(finalResults)

        // 이미지 처리 종료
        imageProxy.close()
    }

    private fun preProcessMultipleCroppedImages(imageProxy: ImageProxy): List<Triple<Bitmap, Int, Int>> {
        val bitmap = imageProxy.toBitmap()

        // 카메라 회전 각도 확인
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees

        // 회전 각도를 적용하여 이미지를 회전시킴
        val rotatedBitmap = bitmap.rotate(rotationDegrees.toFloat())

        // 6개의 구역에 대한 이미지와 각 구역의 (x, y) 오프셋 저장
        return listOf(
            Triple(Bitmap.createBitmap(rotatedBitmap, 0, 0, 640, 640), 0, 0), // 왼쪽 위
            Triple(Bitmap.createBitmap(rotatedBitmap, 440, 0, 640, 640), 440, 0), // 오른쪽 위
            Triple(Bitmap.createBitmap(rotatedBitmap, 0, 440, 640, 640), 0, 440), // 왼쪽 중간
            Triple(Bitmap.createBitmap(rotatedBitmap, 440, 440, 640, 640), 440, 440), // 오른쪽 중간
            Triple(Bitmap.createBitmap(rotatedBitmap, 0, 880, 640, 640), 0, 880), // 왼쪽 아래
            Triple(Bitmap.createBitmap(rotatedBitmap, 440, 880, 640, 640), 440, 880) // 오른쪽 아래
        )
    }

    // 중복된 결과 제거하는 Non-Maximum Suppression (NMS) 적용
    private fun postProcess(rawOutput: OrtSession.Result?, offsetX: Int, offsetY: Int): MutableList<Result> {
        val output = mutableListOf<Result>()
        rawOutput?.run {
            val outputArray = (get(0).value as Array<*>)[0] as Array<*>
            outputArray.asSequence()
                .map { it as FloatArray }
                .filter { it[CONFIDENCE] > confidenceThreshold }
                .mapTo(output) { it.toResultWithOffset(offsetX, offsetY) }
        }

        // NMS 적용하여 중복된 바운딩 박스를 제거
        return applyNMS(output, 0.3f) // IoU 임계값을 0.3로 설정
    }

    // Non-Maximum Suppression (NMS) 함수
    private fun applyNMS(results: MutableList<Result>, iouThreshold: Float): MutableList<Result> {
        val finalResults = mutableListOf<Result>()

        // 결과들을 신뢰도 순으로 내림차순 정렬
        val sortedResults = results.sortedByDescending { it.confidence }

        val isRemoved = BooleanArray(sortedResults.size) { false }

        for (i in sortedResults.indices) {
            if (isRemoved[i]) continue

            val result = sortedResults[i]
            finalResults.add(result)

            // 남은 결과와 비교해서 IoU가 threshold를 넘으면 제거
            for (j in i + 1 until sortedResults.size) {
                if (!isRemoved[j]) {
                    val iou = calculateIoU(result, sortedResults[j])

                    // 겹치는 바운딩 박스를 제거
                    if (iou > iouThreshold) {
                        isRemoved[j] = true
                    }
                }
            }
        }

        return finalResults
    }

    // 두 바운딩 박스의 IoU (Intersection over Union)를 계산 (전체 이미지 좌표계 기준)
    private fun calculateIoU(box1: Result, box2: Result): Float {
        val x1 = max(box1.left, box2.left)
        val y1 = max(box1.top, box2.top)
        val x2 = min(box1.left + box1.width, box2.left + box2.width)
        val y2 = min(box1.top + box1.height, box2.top + box2.height)

        val intersectionArea = max(0f, x2 - x1) * max(0f, y2 - y1)
        val box1Area = box1.width * box1.height
        val box2Area = box2.width * box2.height
        val unionArea = box1Area + box2Area - intersectionArea

        return if (unionArea == 0f) 0f else intersectionArea / unionArea
    }

    // 각 블록의 결과에 오프셋 적용
//    private fun postProcess(rawOutput: OrtSession.Result?, offsetX: Int, offsetY: Int): MutableList<Result> {
//        val output = mutableListOf<Result>()
//        rawOutput?.run {
//            val outputArray = (get(0).value as Array<*>)[0] as Array<*>
//            outputArray.asSequence()
//                .map { it as FloatArray }
//                .filter { it[CONFIDENCE] > confidenceThreshold }
//                .mapTo(output) { it.toResultWithOffset(offsetX, offsetY) } // 오프셋 적용된 좌표로 변환
//        }
//        return output
//    }

    // 오프셋을 적용한 Result 생성
    private fun FloatArray.toResultWithOffset(offsetX: Int, offsetY: Int): Result {

        val phoneOffsetX = phoneWidth / 1080f * offsetX
        val phoneOffsetY = (phoneHeight + diffY) / 1920f * offsetY

        val left = max(0f, this[LEFT] * dx + phoneOffsetX)
        val top = max(0f, this[TOP] * dy + phoneOffsetY - diffY / 2)
        val width = min(width.toFloat(), max(0f, this[RIGHT] - this[LEFT])) * dx
        val height = min(height.toFloat(), max(0f, this[BOTTOM] - this[TOP])) * dy

        println("Converted Box: (${left}, ${top}, ${width}, ${height}) with Offset (${offsetX}, ${offsetY})")

        return Result(
            left = left,
            top = top,
            width = width,
            height = height,
            className = classes[this[CLASS_INDEX].toInt()],
            confidence = this[CONFIDENCE] * 100,
        )
    }


    private fun preProcessBitmap(bitmap: Bitmap): OnnxTensor {
        // 크기를 YOLO 입력 크기에 맞게 다시 스케일링
        val rescaledBitmap = Bitmap.createScaledBitmap(bitmap, width, height, true)

        val cap = shape.reduce { acc, l -> acc * l }.toInt()
        val order = ByteOrder.nativeOrder()
        val buffer = ByteBuffer.allocateDirect(cap * Float.SIZE_BYTES).order(order).asFloatBuffer()
        val area = width * height

        // 이미지 데이터를 Tensor로 변환
        for (i in 0 until width) {
            for (j in 0 until height) {
                val idx = width * i + j
                val pixelValue = rescaledBitmap.getPixel(j, i)

                buffer.put(idx, Color.red(pixelValue) / imageStd)
                buffer.put(idx + area, Color.green(pixelValue) / imageStd)
                buffer.put(idx + area * 2, Color.blue(pixelValue) / imageStd)
            }
        }

        return OnnxTensor.createTensor(ortEnv, buffer, shape)
    }


//    fun infer(imageProxy: ImageProxy) = viewModelScope.launch(Dispatchers.Default) {
//        val inputTensor = preProcessCroppedImage(imageProxy)
//        val rawOutput = process(inputTensor)
//        val output = postProcess(rawOutput)
//
//        _resultList.emit(output)
//        imageProxy.close()
//    }

    private fun cropImageToTopLeft(image: Bitmap, cropSize: Int): Bitmap {
        val croppedWidth = min(cropSize, image.width)
        val croppedHeight = min(cropSize, image.height)

        // 왼쪽 위 영역을 잘라서 반환
        return Bitmap.createBitmap(image, 0, 0, croppedWidth, croppedHeight)
    }

    private fun preProcessCroppedImage(imageProxy: ImageProxy): OnnxTensor {
        val bitmap = imageProxy.toBitmap()

        // 카메라 회전 각도 확인
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees

        // 회전 각도를 적용하여 이미지를 회전시킴
        val rotatedBitmap = bitmap.rotate(rotationDegrees.toFloat())

        // 640x640 크기로 이미지 자르기
        val croppedBitmap = cropImageToTopLeft(rotatedBitmap, 640)

        // 크기를 YOLO 입력 크기에 맞게 다시 스케일링 (필요할 경우)
        val rescaledBitmap = Bitmap.createScaledBitmap(croppedBitmap, width, height, true)

        val cap = shape.reduce { acc, l -> acc * l }.toInt()
        val order = ByteOrder.nativeOrder()
        val buffer = ByteBuffer.allocateDirect(cap * Float.SIZE_BYTES).order(order).asFloatBuffer()
        val area = width * height

        // 이미지 데이터를 Tensor로 변환
        for (i in 0 until width) {
            for (j in 0 until height) {
                val idx = width * i + j
                val pixelValue = rescaledBitmap.getPixel(j, i)

                buffer.put(idx, Color.red(pixelValue) / imageStd)
                buffer.put(idx + area, Color.green(pixelValue) / imageStd)
                buffer.put(idx + area * 2, Color.blue(pixelValue) / imageStd)
            }
        }

        return OnnxTensor.createTensor(ortEnv, buffer, shape)
    }


    // Bitmap을 회전시키는 함수
    fun Bitmap.rotate(degrees: Float): Bitmap {
        val matrix = Matrix()
        matrix.postRotate(degrees)
        return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
    }

    private fun preProcess(imageProxy: ImageProxy): OnnxTensor {
        val bitmap = imageProxy.toBitmap()

        // 카메라 회전 각도 확인
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees

        // 회전 각도를 적용하여 이미지를 회전시킴
        val rotatedBitmap = bitmap.rotate(rotationDegrees.toFloat())

        val rescaledBitmap = Bitmap.createScaledBitmap(rotatedBitmap, width, height, true)

        val cap = shape.reduce { acc, l -> acc * l }.toInt()
        val order = ByteOrder.nativeOrder()
        val buffer = ByteBuffer.allocateDirect(cap * Float.SIZE_BYTES).order(order).asFloatBuffer()
        val area = width * height

        for (i in 0 until width) {
            for (j in 0 until height) {
                val idx = width * i + j
                val pixelValue = rescaledBitmap.getPixel(j, i)

                buffer.put(idx, Color.red(pixelValue) / imageStd)
                buffer.put(idx + area, Color.green(pixelValue) / imageStd)
                buffer.put(idx + area * 2, Color.blue(pixelValue) / imageStd)
            }
        }
        return OnnxTensor.createTensor(ortEnv, buffer, shape)
    }

    private fun process(inputTensor: OnnxTensor): OrtSession.Result? {
        inputTensor.use {
            val inputName = ortSession.inputNames.first()
            return ortSession.run(Collections.singletonMap(inputName, inputTensor))
        }
    }

    private fun postProcess(rawOutput: OrtSession.Result?): MutableList<Result> {
        val output = mutableListOf<Result>()
        rawOutput?.run {
            val outputArray = (get(0).value as Array<*>)[0] as Array<*>
            outputArray.asSequence()
                .map { it as FloatArray }
                .filter { it[CONFIDENCE] > confidenceThreshold }
                .mapTo(output) { it.toResult() }
        }
        return output
    }

    private fun FloatArray.toResult(): Result {
        val left = max(0f, this[LEFT] * dx)
        val top = max(0f, this[TOP] * dy - diffY / 2)
        val width = min(width.toFloat(), max(0f, this[RIGHT] - this[LEFT])) * dx
        val height = min(height.toFloat(), max(0f, this[BOTTOM] - this[TOP])) * dy

        return Result(
            left = left,
            top = top,
            width = width,
            height = height,
            className = classes[this[CLASS_INDEX].toInt()],
            confidence = this[CONFIDENCE] * 100,
        )
    }

    fun setDiff(viewWidth: Int, viewHeight: Int) {
        phoneHeight = viewHeight
        phoneWidth = viewWidth

        diffY = viewWidth * 16f / 9f - viewHeight

        dx = viewWidth / 1080f
        dy = (viewHeight + diffY) / 1920f

//        dx = viewWidth / width.toFloat()
//        dy = dx * 16f / 9f
//        diffY = viewWidth * 16f / 9f - viewHeight
    }

    private fun readModel(): ByteArray =
        resources.openRawResource(R.raw.yolov10n).readBytes()

    private fun readClasses(): List<String> =
        resources.openRawResource(R.raw.classes).bufferedReader().readLines()

    override fun onCleared() {
        ortEnv.close()
        ortSession.close()
        super.onCleared()
    }

    companion object {
        const val LEFT = 0
        const val TOP = 1
        const val RIGHT = 2
        const val BOTTOM = 3
        const val CONFIDENCE = 4
        const val CLASS_INDEX = 5
    }
}