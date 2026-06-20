import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const SETTINGS = [
    {
        id: "LLMPlus.MistralApiKey",
        key: "mistral_api_key",
        name: "Mistral API key",
        type: "text",
        defaultValue: "",
        attrs: { type: "password", autocomplete: "off" },
    },
    {
        id: "LLMPlus.GeminiApiKey",
        key: "gemini_api_key",
        name: "Gemini API key",
        type: "text",
        defaultValue: "",
        attrs: { type: "password", autocomplete: "off" },
    },
    {
        id: "LLMPlus.LMStudioApiBase",
        key: "lmstudio_api_base",
        name: "LM Studio API base",
        type: "text",
        defaultValue: "http://127.0.0.1:1234/v1",
    },
    {
        id: "LLMPlus.LMStudioApiKey",
        key: "lmstudio_api_key",
        name: "LM Studio API key (optional)",
        type: "text",
        defaultValue: "",
        attrs: { type: "password", autocomplete: "off" },
    },
    {
        id: "LLMPlus.ImageMaxSize",
        key: "image_max_size",
        name: "Image maximum side (px)",
        type: "number",
        defaultValue: 768,
        attrs: { min: 64, step: 1 },
    },
    {
        id: "LLMPlus.ImageMaxKb",
        key: "image_max_kb",
        name: "Image maximum JPEG size (KB)",
        type: "number",
        defaultValue: 400,
        attrs: { min: 32, step: 1 },
    },
];

async function saveBackendSettings(values) {
    const response = await api.fetchApi("/llm-plus/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(values),
    });
    if (!response.ok) {
        throw new Error(await response.text());
    }
}

function currentSettings() {
    return Object.fromEntries(
        SETTINGS.map((setting) => [
            setting.key,
            app.ui.settings.getSettingValue(setting.id, setting.defaultValue),
        ]),
    );
}

async function refreshNodeModels(node) {
    const response = await api.fetchApi("/llm-plus/models", { cache: "no-store" });
    const payload = await response.json();
    if (!response.ok || !payload.ok) {
        throw new Error(payload.error || "Could not load LM Studio models.");
    }

    const widget = node.widgets?.find((item) => item.name === "model");
    if (!widget) {
        return;
    }
    widget.options.values = payload.models;
    if (!payload.models.includes(widget.value)) {
        widget.value = payload.models[0];
    }
    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "LLMPlus.Prompt",

    async setup() {
        for (const setting of SETTINGS) {
            app.ui.settings.addSetting({
                id: setting.id,
                name: setting.name,
                category: ["LLM++", "API and image settings", setting.name],
                type: setting.type,
                defaultValue: setting.defaultValue,
                attrs: setting.attrs,
                onChange: (value) => {
                    saveBackendSettings({ [setting.key]: value }).catch((error) => {
                        console.error("LLM++ settings update failed", error);
                    });
                },
            });
        }

        await saveBackendSettings(currentSettings()).catch((error) => {
            console.error("LLM++ initial settings sync failed", error);
        });
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "LLMPlusPrompt") {
            return;
        }

        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            originalCreated?.apply(this, arguments);
            const refreshWidget = this.addWidget("button", "Refresh LM Studio models", null, async () => {
                try {
                    await saveBackendSettings(currentSettings());
                    await refreshNodeModels(this);
                } catch (error) {
                    alert(`LLM++: ${error.message}`);
                }
            });
            const modelIndex = this.widgets?.findIndex((item) => item.name === "model") ?? -1;
            const refreshIndex = this.widgets?.indexOf(refreshWidget) ?? -1;
            if (modelIndex >= 0 && refreshIndex >= 0 && refreshIndex !== modelIndex - 1) {
                this.widgets.splice(refreshIndex, 1);
                this.widgets.splice(modelIndex, 0, refreshWidget);
            }
        };
    },
});
