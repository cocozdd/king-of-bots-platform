package com.kob.botrunningsystem.executor;

import com.kob.botrunningsystem.service.impl.utils.Bot;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

public class JavaRunner implements Runner {
    // 基准限时，单位毫秒
    private final long timeoutMs;

    public JavaRunner(long timeoutMs) {
        this.timeoutMs = timeoutMs;
    }

    private String addUid(String code, String uid) {
        int k = code.indexOf(" implements java.util.function.Supplier<Integer>");
        if (k == -1) {
            return code;  // 未找到标记，直接返回
        }
        return code.substring(0, k) + uid + code.substring(k);
    }

    private ExecutionResult buildError(String type, String errMsg) {
        ExecutionResult result = new ExecutionResult();
        result.setStatus("error");
        result.setErrorType(type);
        result.setStderr(errMsg);
        return result;
    }

    @Override
    public ExecutionResult run(Bot bot) {
        ExecutionResult result = new ExecutionResult();
        long start = System.currentTimeMillis();
        String uid = UUID.randomUUID().toString().substring(0, 8);
        Path tempDir = null;
        try {
            tempDir = Files.createTempDirectory("botrun_");
            // 写入 input
            Files.writeString(tempDir.resolve("input.txt"), bot.getInput(), StandardCharsets.UTF_8);

            // 写入 bot 源码（动态改类名避免冲突）
            String source = addUid(bot.getBotCode(), uid);
            String botClassName = "Bot" + uid;
            Files.writeString(tempDir.resolve(botClassName + ".java"), source, StandardCharsets.UTF_8);

            // 写入包装器，调用 Supplier#get 打印方向
            String runnerClassName = "Runner" + uid;
            String runnerSource = ""
                    + "import java.util.function.Supplier;\n"
                    + "public class " + runnerClassName + " {\n"
                    + "    public static void main(String[] args) throws Exception {\n"
                    + "        Supplier<Integer> bot = new " + botClassName + "();\n"
                    + "        System.out.println(bot.get());\n"
                    + "    }\n"
                    + "}\n";
            Files.writeString(tempDir.resolve(runnerClassName + ".java"), runnerSource, StandardCharsets.UTF_8);

            // 编译
            Process compile = new ProcessBuilder()
                    .command("javac", botClassName + ".java", runnerClassName + ".java")
                    .directory(tempDir.toFile())
                    .start();
            boolean compiled = compile.waitFor(timeoutMs, TimeUnit.MILLISECONDS);
            if (!compiled) {
                compile.destroyForcibly();
                return buildError("timeout", "compile timeout");
            }
            if (compile.exitValue() != 0) {
                result.setStatus("error");
                result.setErrorType("compile");
                result.setExitCode(compile.exitValue());
                result.setStdout(new String(compile.getInputStream().readAllBytes(), StandardCharsets.UTF_8));
                result.setStderr(new String(compile.getErrorStream().readAllBytes(), StandardCharsets.UTF_8));
                result.setTimeMs(System.currentTimeMillis() - start);
                return result;
            }

            // 运行
            Process exec = new ProcessBuilder()
                    .command("java", "-cp", tempDir.toAbsolutePath().toString(), runnerClassName)
                    .directory(tempDir.toFile())
                    .start();
            boolean finished = exec.waitFor(timeoutMs, TimeUnit.MILLISECONDS);
            if (!finished) {
                exec.destroyForcibly();
                return buildError("timeout", "run timeout");
            }
            long end = System.currentTimeMillis();
            result.setExitCode(exec.exitValue());
            result.setTimeMs(end - start);
            result.setStdout(new String(exec.getInputStream().readAllBytes(), StandardCharsets.UTF_8));
            result.setStderr(new String(exec.getErrorStream().readAllBytes(), StandardCharsets.UTF_8));
            if (exec.exitValue() != 0) {
                result.setStatus("error");
                result.setErrorType("runtime");
            } else {
                result.setStatus("success");
                // 解析方向
                String out = result.getStdout().trim();
                try {
                    int dir = Integer.parseInt(out.split("\\s+")[0]);
                    result.setDirection(dir);
                } catch (Exception e) {
                    result.setErrorType("runtime");
                    result.setStatus("error");
                    result.setDirection(null);
                }
            }
            return result;
        } catch (IOException | InterruptedException e) {
            return buildError("unknown", e.getMessage());
        } finally {
            result.setTimeMs(System.currentTimeMillis() - start);
            // 清理临时目录
            if (tempDir != null) {
                try {
                    Files.walk(tempDir)
                            .sorted((p1, p2) -> p2.compareTo(p1))
                            .forEach(p -> {
                                try {
                                    Files.deleteIfExists(p);
                                } catch (IOException ignored) {
                                }
                            });
                } catch (IOException ignored) {
                }
            }
        }
    }
}
