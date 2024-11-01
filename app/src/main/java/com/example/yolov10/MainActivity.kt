package com.example.yolov10

import android.Manifest
import android.os.Bundle
import android.speech.tts.TextToSpeech
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.onSizeChanged
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.yolov10.AppViewModelFactory.FACTORY
import kotlinx.coroutines.delay
import java.util.Locale
import androidx.compose.runtime.LaunchedEffect

class MainActivity : ComponentActivity(), TextToSpeech.OnInitListener {

    private lateinit var tts: TextToSpeech

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // 카메라 권한 요청
        val permissions = arrayOf(Manifest.permission.CAMERA)
        PermissionHelper(this, permissions).launchPermission()

        // TextToSpeech 초기화
        tts = TextToSpeech(this, this)

        setContent {
            val mainViewModel: MainViewModel = viewModel(factory = FACTORY)
            val modifier = Modifier
                .fillMaxSize()
                .onSizeChanged { size ->
                    val width = size.width
                    val height = size.height

                    // 로그로 width와 height 출력
                    println("MAIN resolution: ${width}x${height}")

                    // ViewModel에 화면 크기 전달
                    mainViewModel.setDiff(width, height)
                }

            // 5초마다 결과를 확인하고 음성 알림을 처리
            LaunchedEffect(Unit) {
                var lastResultSpokenTime = 0L  // 마지막으로 알림을 준 시간 저장

                mainViewModel.resultList.collect { resultList ->
                    if (resultList.isNotEmpty()) {
                        val result = resultList[0]

                        // 현재 시간이 마지막 알림 후 5초 지났는지 확인
                        val currentTime = System.currentTimeMillis()
                        if (currentTime - lastResultSpokenTime >= 5000) {
                            if (result.confidence > 80) {
                                val confidencePercentage = result.confidence.toInt()  // 소수점 자리를 버리고 정수로 변환
                                val message = if (result.className == "redlight") {
                                    "빨간불, $confidencePercentage% 입니다."
                                } else if (result.className == "greenlight") {
                                    "초록불, $confidencePercentage% 입니다."
                                } else {
                                    null
                                }

                                // 음성 알림 비동기 처리
                                // 메시지가 null이 아닐 때만 음성 알림 처리
                                message?.let {
                                    speak(it)
                                }

                                // 마지막 알림 시간을 업데이트
                                lastResultSpokenTime = currentTime

                                // 5초 대기
                                delay(5000)
                            }
                        }
                    }
                }
            }

            // 카메라 및 캔버스 뷰 설정
            CameraView(modifier = modifier, viewModel = mainViewModel)
            CanvasView(modifier = modifier, viewModel = mainViewModel)
        }
    }

    // TextToSpeech 초기화 완료 시 호출되는 함수
    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            // TTS 언어 설정
            tts.language = Locale.KOREAN
        }
    }

    private fun speak(text: String) {
        // 음성 알림 비동기 처리
        tts.speak(text, TextToSpeech.QUEUE_ADD, null, null)
    }

    override fun onDestroy() {
        super.onDestroy()
        // TextToSpeech 리소스 해제
        if (::tts.isInitialized) {
            tts.shutdown()
        }
    }
}
