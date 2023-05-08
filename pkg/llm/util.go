package llm

import (
	"bytes"
	"strings"
	"text/template"
)

func matchAnyStop(output string, stops []string) (bool, string) {
	for _, stop := range stops {
		if strings.HasSuffix(output, stop) {
			return true, stop
		}
	}
	return false, ""
}

func render(promptTemplate *template.Template, input any) (string, error) {
	buff := &bytes.Buffer{}
	err := promptTemplate.Execute(buff, map[string]any{"Input": input})
	if err != nil {
		return "", err
	}
	renderedPrompt := buff.String()
	return renderedPrompt, nil
}
