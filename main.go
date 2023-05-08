package main

import (
	"flag"

	"k8s.io/klog/v2"

	"github.com/bdqfork/go-llama.cpp/pkg/config"
	"github.com/bdqfork/go-llama.cpp/pkg/context"
	"github.com/bdqfork/go-llama.cpp/pkg/server"
)

func main() {
	flagSet := flag.CommandLine
	klog.InitFlags(flagSet)

	c := config.New()
	c.Init(flagSet)

	flag.Parse()

	c.Print()
	c.LoadModelConfigs()

	ctx := context.New(c)
	defer ctx.Close()

	s := server.New(ctx)
	s.Run()
}
