const { defineConfig } = require('@vue/cli-service')
module.exports = defineConfig({
  transpileDependencies: true,
  devServer: {
    proxy: {
      '/api': {
        target: 'http://localhost:3000',
        changeOrigin: true
      },
      '/ai': {
        target: 'http://localhost:3000',
        changeOrigin: true,
        // SSE 流式传输需要的配置
        onProxyRes: function(proxyRes, req, res) {
          // 禁用响应缓冲，确保SSE实时传输
          proxyRes.headers['cache-control'] = 'no-cache';
          proxyRes.headers['x-accel-buffering'] = 'no';
        }
      }
    },
    // 禁用压缩，SSE不兼容压缩
    compress: false
  }
})
