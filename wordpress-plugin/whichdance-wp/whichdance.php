<?php
/**
 * Plugin Name: whichdance
 * Description: Upload a tune, get a guess at which traditional folk dance it is (via the whichdance inference service). Thin client only — proxies to an external FastAPI service, no ML runs in PHP.
 * Version: 0.1.0
 * Author: whichdance
 * License: GPL-2.0-or-later
 */

if (!defined('ABSPATH')) {
    exit; // no direct access
}

define('WHICHDANCE_VERSION', '0.1.0');

/**
 * Settings: API base URL, configurable under Settings > whichdance.
 * Defaults to a local service on the same host.
 */
function whichdance_api_base_url() {
    $url = get_option('whichdance_api_base_url', 'http://127.0.0.1:8000');
    return untrailingslashit($url);
}

add_action('admin_menu', function () {
    add_options_page(
        'whichdance',
        'whichdance',
        'manage_options',
        'whichdance',
        'whichdance_settings_page'
    );
});

add_action('admin_init', function () {
    register_setting('whichdance', 'whichdance_api_base_url', [
        'type' => 'string',
        'sanitize_callback' => 'esc_url_raw',
        'default' => 'http://127.0.0.1:8000',
    ]);
});

function whichdance_settings_page() {
    ?>
    <div class="wrap">
        <h1>whichdance settings</h1>
        <form method="post" action="options.php">
            <?php settings_fields('whichdance'); ?>
            <table class="form-table">
                <tr>
                    <th scope="row"><label for="whichdance_api_base_url">Inference service URL</label></th>
                    <td>
                        <input type="url" id="whichdance_api_base_url" name="whichdance_api_base_url"
                               value="<?php echo esc_attr(get_option('whichdance_api_base_url', 'http://127.0.0.1:8000')); ?>"
                               class="regular-text" placeholder="http://127.0.0.1:8000">
                        <p class="description">
                            Base URL of the whichdance FastAPI service (its <code>/predict</code> endpoint).
                            Usually an internal address — WordPress proxies to it server-side, so it doesn't need to be publicly reachable.
                        </p>
                    </td>
                </tr>
            </table>
            <?php submit_button(); ?>
        </form>
    </div>
    <?php
}

/**
 * REST route: browser -> WordPress (same-origin) -> whichdance service.
 * Keeps the inference service off the public internet and avoids CORS
 * entirely on the frontend.
 */
add_action('rest_api_init', function () {
    register_rest_route('whichdance/v1', '/predict', [
        'methods' => 'POST',
        'callback' => 'whichdance_handle_predict',
        'permission_callback' => '__return_true', // public: anyone who can view the page can use the widget
    ]);
});

function whichdance_handle_predict(WP_REST_Request $request) {
    $files = $request->get_file_params();
    if (empty($files['file']) || $files['file']['error'] !== UPLOAD_ERR_OK) {
        return new WP_Error('whichdance_no_file', 'No audio file uploaded.', ['status' => 400]);
    }
    $upload = $files['file'];

    $boundary = wp_generate_password(24, false);
    $body = whichdance_build_multipart_body($upload, $boundary);

    $response = wp_remote_post(whichdance_api_base_url() . '/predict', [
        'timeout' => 60,
        'headers' => ['Content-Type' => 'multipart/form-data; boundary=' . $boundary],
        'body' => $body,
    ]);

    if (is_wp_error($response)) {
        return new WP_Error('whichdance_upstream_error', $response->get_error_message(), ['status' => 502]);
    }

    $code = wp_remote_retrieve_response_code($response);
    $payload = json_decode(wp_remote_retrieve_body($response), true);

    if ($code !== 200) {
        $detail = is_array($payload) && isset($payload['detail']) ? $payload['detail'] : 'Inference service error.';
        return new WP_Error('whichdance_service_error', $detail, ['status' => $code ?: 502]);
    }

    return rest_ensure_response($payload);
}

/**
 * Build a multipart/form-data body from a PHP $_FILES-style upload array,
 * since wp_remote_post doesn't support file uploads natively.
 */
function whichdance_build_multipart_body($upload, $boundary) {
    $contents = file_get_contents($upload['tmp_name']);
    $filename = basename($upload['name']);
    $mimetype = $upload['type'] ?: 'application/octet-stream';

    $body = "--{$boundary}\r\n";
    $body .= "Content-Disposition: form-data; name=\"file\"; filename=\"{$filename}\"\r\n";
    $body .= "Content-Type: {$mimetype}\r\n\r\n";
    $body .= $contents . "\r\n";
    $body .= "--{$boundary}--\r\n";

    return $body;
}

/**
 * Shortcode: [whichdance]
 */
add_shortcode('whichdance', function () {
    wp_enqueue_script(
        'whichdance-predict',
        plugins_url('assets/predict.js', __FILE__),
        [],
        WHICHDANCE_VERSION,
        true
    );
    wp_enqueue_style(
        'whichdance-style',
        plugins_url('assets/style.css', __FILE__),
        [],
        WHICHDANCE_VERSION
    );
    wp_localize_script('whichdance-predict', 'whichdanceConfig', [
        'restUrl' => esc_url_raw(rest_url('whichdance/v1/predict')),
        'nonce' => wp_create_nonce('wp_rest'),
    ]);

    ob_start();
    ?>
    <div class="whichdance-widget">
        <input type="file" class="whichdance-file" accept="audio/*">
        <div class="whichdance-result"></div>
    </div>
    <?php
    return ob_get_clean();
});
