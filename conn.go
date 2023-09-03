package yolotriton

import (
	"context"
	"time"

	triton "github.com/dev6699/yolotriton/grpc-client"
)

func ServerLiveRequest(client triton.GRPCInferenceServiceClient) (*triton.ServerLiveResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	serverLiveRequest := triton.ServerLiveRequest{}
	serverLiveResponse, err := client.ServerLive(ctx, &serverLiveRequest)
	if err != nil {
		return nil, err
	}
	return serverLiveResponse, nil
}

func ServerReadyRequest(client triton.GRPCInferenceServiceClient) (*triton.ServerReadyResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	serverReadyRequest := triton.ServerReadyRequest{}
	serverReadyResponse, err := client.ServerReady(ctx, &serverReadyRequest)
	if err != nil {
		return nil, err
	}
	return serverReadyResponse, nil
}

func ModelMetadataRequest(client triton.GRPCInferenceServiceClient, modelName string, modelVersion string) (*triton.ModelMetadataResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	modelMetadataRequest := triton.ModelMetadataRequest{
		Name:    modelName,
		Version: modelVersion,
	}
	modelMetadataResponse, err := client.ModelMetadata(ctx, &modelMetadataRequest)
	if err != nil {
		return nil, err
	}
	return modelMetadataResponse, nil
}

func ModelInferRequest(client triton.GRPCInferenceServiceClient, modelInferRequest *triton.ModelInferRequest) (*triton.ModelInferResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	modelInferResponse, err := client.ModelInfer(ctx, modelInferRequest)
	if err != nil {
		return nil, err
	}

	return modelInferResponse, nil
}
