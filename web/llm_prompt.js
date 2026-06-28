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

async function fetchJson(path, options = {}) {
    const response = await api.fetchApi(path, options);
    const payload = await response.json();
    if (!response.ok || !payload.ok) {
        throw new Error(payload.error || "LLM++ request failed.");
    }
    return payload;
}

async function loadPresets() {
    const payload = await fetchJson("/llm-plus/presets", { cache: "no-store" });
    return payload.presets || [];
}

async function loadPresetText(name) {
    const payload = await fetchJson("/llm-plus/presets/get", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
    });
    return payload.preset;
}

async function savePresetText(name, text, originalName = "") {
    return fetchJson("/llm-plus/presets/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, text, original_name: originalName }),
    });
}

async function deletePresetText(name) {
    return fetchJson("/llm-plus/presets/delete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
    });
}

function setNodePrompt(node, text) {
    const promptWidget = node.widgets?.find((item) => item.name === "prompt");
    if (!promptWidget) {
        return;
    }
    promptWidget.value = text || "";
    promptWidget.callback?.(promptWidget.value);
    node.setDirtyCanvas(true, true);
}

function ensurePresetDialogStyles() {
    if (document.getElementById("llm-plus-preset-dialog-styles")) {
        return;
    }
    const style = document.createElement("style");
    style.id = "llm-plus-preset-dialog-styles";
    style.textContent = `
.llm-plus-preset-backdrop {
    position: fixed;
    inset: 0;
    z-index: 10000;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(0, 0, 0, 0.55);
}
.llm-plus-preset-dialog {
    box-sizing: border-box;
    width: min(760px, calc(100vw - 32px));
    max-height: calc(100vh - 48px);
    padding: 16px;
    overflow: auto;
    color: var(--fg-color, #ddd);
    background: var(--comfy-menu-bg, #222);
    border: 1px solid var(--border-color, #555);
    border-radius: 8px;
    box-shadow: 0 18px 60px rgba(0, 0, 0, 0.45);
}
.llm-plus-preset-dialog h3 {
    margin: 0 0 12px;
    font-size: 16px;
    font-weight: 600;
}
.llm-plus-preset-dialog label {
    display: grid;
    gap: 5px;
    margin: 10px 0;
    font-size: 12px;
}
.llm-plus-preset-dialog select,
.llm-plus-preset-dialog input,
.llm-plus-preset-dialog textarea {
    box-sizing: border-box;
    width: 100%;
    color: var(--input-text, inherit);
    background: var(--comfy-input-bg, #111);
    border: 1px solid var(--border-color, #555);
    border-radius: 6px;
}
.llm-plus-preset-dialog select,
.llm-plus-preset-dialog input {
    min-height: 32px;
    padding: 4px 8px;
}
.llm-plus-preset-dialog textarea {
    min-height: 220px;
    padding: 8px;
    resize: vertical;
    font-family: inherit;
}
.llm-plus-preset-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 12px;
}
.llm-plus-preset-actions button {
    min-height: 32px;
    padding: 4px 12px;
    color: inherit;
    background: var(--comfy-button-bg, #333);
    border: 1px solid var(--border-color, #555);
    border-radius: 6px;
    cursor: pointer;
}
.llm-plus-preset-status {
    min-height: 18px;
    margin-top: 10px;
    font-size: 12px;
    opacity: 0.9;
}
`;
    document.head.appendChild(style);
}

function uniqueCopyName(baseName, names) {
    const taken = new Set(names);
    let index = 1;
    let candidate = `${baseName} copy`;
    while (taken.has(candidate)) {
        index += 1;
        candidate = `${baseName} copy ${index}`;
    }
    return candidate;
}

function createPresetEditor(node, state) {
    ensurePresetDialogStyles();

    const backdrop = document.createElement("div");
    backdrop.className = "llm-plus-preset-backdrop";

    const dialog = document.createElement("div");
    dialog.className = "llm-plus-preset-dialog";
    dialog.innerHTML = `
        <h3>Preset Editor</h3>
        <label>
            Preset
            <select data-field="select"></select>
        </label>
        <label>
            Name
            <input data-field="name" type="text" autocomplete="off" />
        </label>
        <label>
            Text
            <textarea data-field="text" spellcheck="false"></textarea>
        </label>
        <div class="llm-plus-preset-actions">
            <button data-action="new" type="button">New</button>
            <button data-action="save" type="button">Save</button>
            <button data-action="duplicate" type="button">Duplicate</button>
            <button data-action="delete" type="button">Delete</button>
            <button data-action="close" type="button">Close</button>
        </div>
        <div class="llm-plus-preset-status" data-field="status"></div>
    `;
    backdrop.appendChild(dialog);
    document.body.appendChild(backdrop);

    const select = dialog.querySelector('[data-field="select"]');
    const nameInput = dialog.querySelector('[data-field="name"]');
    const textInput = dialog.querySelector('[data-field="text"]');
    const status = dialog.querySelector('[data-field="status"]');
    let originalName = "";

    const setStatus = (message) => {
        status.textContent = message || "";
    };

    const refreshSelect = (selectedName = "") => {
        select.replaceChildren();
        for (const preset of state.presets) {
            const option = document.createElement("option");
            option.value = preset.name;
            option.textContent = preset.name;
            select.appendChild(option);
        }
        if (selectedName && state.presets.some((preset) => preset.name === selectedName)) {
            select.value = selectedName;
        }
    };

    const loadIntoEditor = async (presetName) => {
        if (!presetName) {
            originalName = "";
            nameInput.value = "";
            textInput.value = "";
            setStatus("New preset will be created.");
            return;
        }
        const preset = await loadPresetText(presetName);
        originalName = preset.name;
        nameInput.value = preset.name;
        textInput.value = preset.text || "";
        setStatus("");
    };

    const syncNodePresetWidget = (selectedName) => {
        const presetWidget = node.widgets?.find((item) => item.name === "Preset");
        if (!presetWidget) {
            return;
        }
        const names = state.presets.map((preset) => preset.name);
        presetWidget.options.values = names.length ? names : ["No presets"];
        presetWidget.value = selectedName || names[0] || "No presets";
        node.setDirtyCanvas(true, true);
    };

    refreshSelect(state.selectedName);
    loadIntoEditor(state.selectedName || state.presets[0]?.name || "").catch((error) => {
        setStatus(error.message);
    });

    select.addEventListener("change", () => {
        loadIntoEditor(select.value).catch((error) => {
            setStatus(error.message);
        });
    });

    dialog.querySelector('[data-action="new"]').addEventListener("click", () => {
        originalName = "";
        select.value = "";
        nameInput.value = "";
        textInput.value = "";
        nameInput.focus();
        setStatus("New preset will be created.");
    });

    dialog.querySelector('[data-action="duplicate"]').addEventListener("click", () => {
        const names = state.presets.map((preset) => preset.name);
        const sourceName = nameInput.value.trim() || state.selectedName || "Preset";
        originalName = "";
        nameInput.value = uniqueCopyName(sourceName, names);
        nameInput.focus();
        nameInput.select();
        setStatus("Copy will be created when you save.");
    });

    dialog.querySelector('[data-action="save"]').addEventListener("click", async () => {
        try {
            const name = nameInput.value.trim();
            const result = await savePresetText(name, textInput.value, originalName);
            state.presets = result.presets || [];
            state.selectedName = result.name;
            originalName = result.name;
            refreshSelect(result.name);
            syncNodePresetWidget(result.name);
            setNodePrompt(node, textInput.value);
            setStatus(`${result.action || "saved"} preset "${result.name}".`);
        } catch (error) {
            setStatus(error.message);
        }
    });

    dialog.querySelector('[data-action="delete"]').addEventListener("click", async () => {
        const name = (originalName || nameInput.value).trim();
        if (!name || !confirm(`Delete preset "${name}"?`)) {
            return;
        }
        try {
            const result = await deletePresetText(name);
            state.presets = result.presets || [];
            state.selectedName = state.presets[0]?.name || "";
            refreshSelect(state.selectedName);
            syncNodePresetWidget(state.selectedName);
            await loadIntoEditor(state.selectedName);
            setStatus(`Deleted preset "${name}".`);
        } catch (error) {
            setStatus(error.message);
        }
    });

    const close = () => {
        backdrop.remove();
    };
    dialog.querySelector('[data-action="close"]').addEventListener("click", close);
    backdrop.addEventListener("click", (event) => {
        if (event.target === backdrop) {
            close();
        }
    });
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
            const presetState = {
                presets: [],
                selectedName: "",
            };

            const presetWidget = this.addWidget("combo", "Preset", "Loading...", async (value) => {
                if (!value || value === "Loading..." || value === "No presets") {
                    return;
                }
                try {
                    const preset = await loadPresetText(value);
                    presetState.selectedName = preset.name;
                    setNodePrompt(this, preset.text);
                } catch (error) {
                    alert(`LLM++: ${error.message}`);
                }
            }, { values: ["Loading..."] });

            const editPresetWidget = this.addWidget("button", "Edit presets", null, async () => {
                try {
                    presetState.presets = await loadPresets();
                    if (!presetState.selectedName && presetState.presets.length) {
                        presetState.selectedName = presetState.presets[0].name;
                    }
                    createPresetEditor(this, presetState);
                } catch (error) {
                    alert(`LLM++: ${error.message}`);
                }
            });

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

            const promptIndex = this.widgets?.findIndex((item) => item.name === "prompt") ?? -1;
            const presetIndex = this.widgets?.indexOf(presetWidget) ?? -1;
            const editPresetIndex = this.widgets?.indexOf(editPresetWidget) ?? -1;
            if (promptIndex >= 0 && presetIndex >= 0 && editPresetIndex >= 0) {
                this.widgets.splice(Math.max(presetIndex, editPresetIndex), 1);
                this.widgets.splice(Math.min(presetIndex, editPresetIndex), 1);
                this.widgets.splice(promptIndex, 0, presetWidget, editPresetWidget);
            }

            loadPresets().then((presets) => {
                presetState.presets = presets;
                const names = presets.map((preset) => preset.name);
                presetWidget.options.values = names.length ? names : ["No presets"];
                presetWidget.value = names[0] || "No presets";
                presetState.selectedName = names[0] || "";
                this.setDirtyCanvas(true, true);
            }).catch((error) => {
                presetWidget.options.values = ["No presets"];
                presetWidget.value = "No presets";
                console.error("LLM++ preset load failed", error);
                this.setDirtyCanvas(true, true);
            });
        };
    },
});
