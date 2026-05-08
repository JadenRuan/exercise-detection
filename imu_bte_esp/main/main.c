#include "driver/i2c.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "host/ble_hs.h"
#include "host/ble_uuid.h"
#include "host/util/util.h"
#include "mpu6050.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "nvs_flash.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"
#include <stdio.h>
#include <string.h>

/* ── I2C / MPU6050 ────────────────────────────────────────────────── */
#define I2C_MASTER_SCL_IO   19
#define I2C_MASTER_SDA_IO   18
#define I2C_MASTER_NUM      I2C_NUM_0
#define I2C_MASTER_FREQ_HZ  400000
#define SAMPLE_PERIOD_MS    10

/* ── BLE ──────────────────────────────────────────────────────────── */
#define DEVICE_NAME         "ESP32_IMU"

/*
 * 128-bit UUIDs — keep these in sync with your Python script.
 * Service: 12345678-1234-1234-1234-123456789ABC
 * Char:    12345678-1234-1234-1234-123456789ABD
 */
static const ble_uuid128_t service_uuid =
    BLE_UUID128_INIT(0xBC,0x9A,0x78,0x56,0x34,0x12,0x34,0x12,
                     0x34,0x12,0x34,0x12,0x78,0x56,0x34,0x12);

static const ble_uuid128_t char_uuid =
    BLE_UUID128_INIT(0xBD,0x9A,0x78,0x56,0x34,0x12,0x34,0x12,
                     0x34,0x12,0x34,0x12,0x78,0x56,0x34,0x12);

static const char *TAG        = "mpu6050_nimble";
static mpu6050_handle_t mpu6050 = NULL;

/* Connection state */
static uint16_t g_conn_handle   = BLE_HS_CONN_HANDLE_NONE;
static uint16_t g_attr_handle   = 0;   /* populated after GATT registration */
static bool     g_notify_enabled = false;

/* ── GATT characteristic access callback ──────────────────────────── */
/*
 * NimBLE calls this when the client reads the characteristic OR
 * writes to the CCCD (enabling/disabling notifications).
 * We only care about the subscription event here.
 */
static int imu_char_access_cb(uint16_t conn_handle, uint16_t attr_handle,
                               struct ble_gatt_access_ctxt *ctxt, void *arg)
{
    /* We expose a notify-only characteristic; reads return 0 bytes. */
    if (ctxt->op == BLE_GATT_ACCESS_OP_READ_CHR) {
        return 0;
    }
    return BLE_ATT_ERR_UNLIKELY;
}

/* ── GATT service table ───────────────────────────────────────────── */
/*
 * NimBLE's declarative table approach:
 *   - One service with one notify characteristic.
 *   - NimBLE automatically adds the CCCD descriptor.
 */
static const struct ble_gatt_svc_def gatt_svcs[] = {
    {
        .type            = BLE_GATT_SVC_TYPE_PRIMARY,
        .uuid            = &service_uuid.u,
        .characteristics = (struct ble_gatt_chr_def[]) {
            {
                .uuid       = &char_uuid.u,
                .access_cb  = imu_char_access_cb,
                /* BLE_GATT_CHR_F_NOTIFY causes NimBLE to add the CCCD */
                .flags      = BLE_GATT_CHR_F_READ | BLE_GATT_CHR_F_NOTIFY,
                .val_handle = &g_attr_handle,
            },
            { 0 }   /* terminate characteristics array */
        },
    },
    { 0 }           /* terminate services array */
};

/* ── GAP event handler ────────────────────────────────────────────── */
static int gap_event_handler(struct ble_gap_event *event, void *arg)
{
    switch (event->type) {

    case BLE_GAP_EVENT_CONNECT:
        if (event->connect.status == 0) {
            g_conn_handle = event->connect.conn_handle;
            ESP_LOGI(TAG, "Client connected, handle=%d", g_conn_handle);
        } else {
            /* Connection failed — restart advertising */
            ESP_LOGW(TAG, "Connect failed, status=%d", event->connect.status);
            g_conn_handle = BLE_HS_CONN_HANDLE_NONE;
            nimble_port_freertos_init(NULL);  /* restart adv via sync cb */
        }
        break;

    case BLE_GAP_EVENT_DISCONNECT:
        ESP_LOGI(TAG, "Client disconnected, reason=%d",
                 event->disconnect.reason);
        g_conn_handle    = BLE_HS_CONN_HANDLE_NONE;
        g_notify_enabled = false;
        /* Restart advertising so a new client can connect */
        ble_gap_adv_start(BLE_OWN_ADDR_PUBLIC, NULL, BLE_HS_FOREVER,
                          &(struct ble_gap_adv_params){
                              .conn_mode = BLE_GAP_CONN_MODE_UND,
                              .disc_mode = BLE_GAP_DISC_MODE_GEN,
                          },
                          gap_event_handler, NULL);
        break;

    case BLE_GAP_EVENT_SUBSCRIBE:
        /*
         * The client toggled our characteristic's CCCD.
         * cur_notify == 1 means notifications are now enabled.
         */
        if (event->subscribe.attr_handle == g_attr_handle) {
            g_notify_enabled = (event->subscribe.cur_notify == 1);
            ESP_LOGI(TAG, "Notifications %s",
                     g_notify_enabled ? "enabled" : "disabled");
        }
        break;

    case BLE_GAP_EVENT_MTU:
        ESP_LOGI(TAG, "MTU updated: conn=%d mtu=%d",
                 event->mtu.conn_handle, event->mtu.value);
        break;

    default:
        break;
    }
    return 0;
}

/* ── Start BLE advertising ────────────────────────────────────────── */
static void ble_advertise(void)
{
    /* Advertising packet — just flags and name (short, fits in 31 bytes) */
    struct ble_hs_adv_fields fields = {0};
    fields.flags            = BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP;
    fields.name             = (uint8_t *)DEVICE_NAME;
    fields.name_len         = strlen(DEVICE_NAME);
    fields.name_is_complete = 1;

    int rc = ble_gap_adv_set_fields(&fields);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_gap_adv_set_fields error: %d", rc);
        return;
    }

    /* Scan response — put the 128-bit UUID here instead */
    struct ble_hs_adv_fields rsp_fields = {0};
    rsp_fields.uuids128             = &service_uuid;
    rsp_fields.num_uuids128         = 1;
    rsp_fields.uuids128_is_complete = 1;

    rc = ble_gap_adv_set_fields(&rsp_fields);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_gap_adv_set_fields error: %d", rc);
        return;
    }

    struct ble_gap_adv_params adv_params = {
        .conn_mode = BLE_GAP_CONN_MODE_UND,
        .disc_mode = BLE_GAP_DISC_MODE_GEN,
    };

    rc = ble_gap_adv_start(BLE_OWN_ADDR_PUBLIC, NULL, BLE_HS_FOREVER,
                           &adv_params, gap_event_handler, NULL);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_gap_adv_start error: %d", rc);
    } else {
        ESP_LOGI(TAG, "Advertising as \"%s\"", DEVICE_NAME);
    }
}

/* ── NimBLE host sync callback ────────────────────────────────────── */
/*
 * Called by the NimBLE host once it has synced with the controller
 * and is ready to use. This is the correct place to start advertising.
 */
static void ble_on_sync(void)
{
    ble_hs_util_ensure_addr(0);   /* ensure we have a valid public address */
    ble_advertise();
}

/* ── NimBLE host task ─────────────────────────────────────────────── */
/* Runs the NimBLE host event loop on its own FreeRTOS task. */
static void nimble_host_task(void *param)
{
    ESP_LOGI(TAG, "NimBLE host task started");
    nimble_port_run();              /* blocks until nimble_port_stop() */
    nimble_port_freertos_deinit();
}

/* ── NimBLE init ──────────────────────────────────────────────────── */
static void ble_init(void)
{
    /* NVS required by BT controller */
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
        ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase();
        nvs_flash_init();
    }

    nimble_port_init();

    /* Register GATT services */
    ble_svc_gap_init();
    ble_svc_gatt_init();
    ble_gatts_count_cfg(gatt_svcs);
    ble_gatts_add_svcs(gatt_svcs);

    /* Set device name visible in GAP */
    ble_svc_gap_device_name_set(DEVICE_NAME);

    /* Register the sync callback — advertising starts from here */
    ble_hs_cfg.sync_cb = ble_on_sync;

    /* Kick off the NimBLE host on its own task */
    nimble_port_freertos_init(nimble_host_task);
}

/* ── I2C / MPU6050 (unchanged from original) ─────────────────────── */
static void i2c_bus_init(void)
{
    i2c_config_t conf = {
        .mode             = I2C_MODE_MASTER,
        .sda_io_num       = (gpio_num_t)I2C_MASTER_SDA_IO,
        .sda_pullup_en    = GPIO_PULLUP_ENABLE,
        .scl_io_num       = (gpio_num_t)I2C_MASTER_SCL_IO,
        .scl_pullup_en    = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_MASTER_FREQ_HZ,
        .clk_flags        = I2C_SCLK_SRC_FLAG_FOR_NOMAL,
    };
    esp_err_t ret = i2c_param_config(I2C_MASTER_NUM, &conf);
    if (ret != ESP_OK) { ESP_LOGE(TAG, "I2C config error"); return; }
    ret = i2c_driver_install(I2C_MASTER_NUM, conf.mode, 0, 0, 0);
    if (ret != ESP_OK) { ESP_LOGE(TAG, "I2C install error"); return; }
}

static void i2c_sensor_mpu6050_init(void)
{
    i2c_bus_init();
    mpu6050 = mpu6050_create(I2C_MASTER_NUM, MPU6050_I2C_ADDRESS);
    if (!mpu6050) { ESP_LOGE(TAG, "MPU6050 create NULL"); return; }
    esp_err_t ret = mpu6050_config(mpu6050, ACCE_FS_4G, GYRO_FS_500DPS);
    if (ret != ESP_OK) { ESP_LOGE(TAG, "MPU6050 config error"); return; }
    ret = mpu6050_wake_up(mpu6050);
    if (ret != ESP_OK) { ESP_LOGE(TAG, "MPU6050 wake error"); return; }
}

/* ── Main ─────────────────────────────────────────────────────────── */
void app_main(void)
{
    i2c_sensor_mpu6050_init();
    ble_init();

    uint8_t dev_id = 0;
    if (mpu6050_get_deviceid(mpu6050, &dev_id) == ESP_OK)
        ESP_LOGI(TAG, "MPU6050 device ID: 0x%02X", dev_id);

    ESP_LOGI(TAG, "Streaming IMU data at 100 Hz over BLE ...");

    char buf[80];
    TickType_t last_wake = xTaskGetTickCount();
    const TickType_t period = pdMS_TO_TICKS(SAMPLE_PERIOD_MS);

    while (1) {
        mpu6050_acce_value_t acce = {0};
        mpu6050_gyro_value_t gyro = {0};
        mpu6050_temp_value_t temp = {0};

        mpu6050_get_acce(mpu6050, &acce);
        mpu6050_get_gyro(mpu6050, &gyro);
        mpu6050_get_temp(mpu6050, &temp);

        int len = snprintf(buf, sizeof(buf),
            "AX:%.3f AY:%.3f AZ:%.3f | GX:%.3f GY:%.3f GZ:%.3f | T:%.2f C",
            acce.acce_x, acce.acce_y, acce.acce_z,
            gyro.gyro_x, gyro.gyro_y, gyro.gyro_z,
            temp.temp);

        ESP_LOGI(TAG, "%s", buf);

        /* Send BLE notification if a client is subscribed */
        if (g_notify_enabled &&
            g_conn_handle != BLE_HS_CONN_HANDLE_NONE) {

            struct os_mbuf *om = ble_hs_mbuf_from_flat(buf, (uint16_t)len);
            if (om) {
                ble_gatts_notify_custom(g_conn_handle, g_attr_handle, om);
            }
        }

        vTaskDelayUntil(&last_wake, period);
    }
}