export const emitUnauthorized = () => {
    if (typeof window !== "undefined") {
        window.dispatchEvent(new CustomEvent("auth:unauthorized"));
    }
};
