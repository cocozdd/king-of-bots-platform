# 多阶段构建 - 支持构建不同模块
FROM maven:3.8-openjdk-17-slim AS builder

WORKDIR /app

# 复制pom文件，利用Docker缓存
COPY pom.xml .
COPY backend/pom.xml backend/
COPY matchingsystem/pom.xml matchingsystem/
COPY botrunningsystem/pom.xml botrunningsystem/

# 下载依赖（利用缓存层）
RUN mvn dependency:go-offline -B

# 复制源代码
COPY backend/src backend/src
COPY matchingsystem/src matchingsystem/src
COPY botrunningsystem/src botrunningsystem/src

# 构建参数 - 指定要构建的模块
ARG MODULE=backend

# 构建指定模块
RUN mvn clean package -pl ${MODULE} -am -DskipTests

# 运行阶段 (使用支持 ARM64 的镜像)
FROM eclipse-temurin:17-jre-jammy

WORKDIR /app

ARG MODULE=backend

# 从构建阶段复制jar包
COPY --from=builder /app/${MODULE}/target/*.jar app.jar

# 健康检查端点
EXPOSE 3000 3001 3002

# JVM优化参数
ENV JAVA_OPTS="-Xms256m -Xmx512m -XX:+UseG1GC -XX:+HeapDumpOnOutOfMemoryError"

ENTRYPOINT ["sh", "-c", "java $JAVA_OPTS -jar app.jar"]
