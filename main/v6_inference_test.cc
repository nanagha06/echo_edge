#include <stdint.h>
#include "esp_log.h"
#include "echoedge_v6_model.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
static const char *TAG = "ECHOEDGE_V6";
constexpr size_t TENSOR_ARENA_SIZE = 160 * 1024;
static uint8_t tensor_arena[TENSOR_ARENA_SIZE];
extern "C" void app_main(void) {
    ESP_LOGI(TAG, "EchoEdge V6 INT8 inference test");
    const tflite::Model *model = tflite::GetModel(g_echoedge_v6_model);
    if (model == nullptr) { ESP_LOGE(TAG, "Model NULL"); return; }
    tflite::MicroMutableOpResolver<6> resolver;
    if (resolver.AddConv2D() != kTfLiteOk || resolver.AddDepthwiseConv2D() != kTfLiteOk || resolver.AddMaxPool2D() != kTfLiteOk || resolver.AddMean() != kTfLiteOk || resolver.AddFullyConnected() != kTfLiteOk || resolver.AddLogistic() != kTfLiteOk) { ESP_LOGE(TAG, "Operator registration failed"); return; }
    tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    if (interpreter.AllocateTensors() != kTfLiteOk) { ESP_LOGE(TAG, "AllocateTensors failed"); return; }
    TfLiteTensor *input = interpreter.input(0);
    TfLiteTensor *output = interpreter.output(0);
    ESP_LOGI(TAG, "Input type=%d shape=%d,%d,%d,%d scale=%f zero=%d", input->type, input->dims->data[0], input->dims->data[1], input->dims->data[2], input->dims->data[3], input->params.scale, input->params.zero_point);
    ESP_LOGI(TAG, "Output type=%d shape=%d,%d scale=%f zero=%d", output->type, output->dims->data[0], output->dims->data[1], output->params.scale, output->params.zero_point);
    for (int i = 0; i < 40 * 101; ++i) input->data.int8[i] = -26;
    if (interpreter.Invoke() != kTfLiteOk) { ESP_LOGE(TAG, "Invoke failed"); return; }
    int8_t q = output->data.int8[0];
    float p = (q - output->params.zero_point) * output->params.scale;
    ESP_LOGI(TAG, "Output q=%d probability=%.6f", q, p);
    ESP_LOGI(TAG, "V6 inference test PASSED");
}
